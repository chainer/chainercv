# Mofidied by:
# Copyright (c) 2016 Shunta Saito

# Original work by:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

import numpy as np

from bbox import bbox_overlaps
from bbox_transform import bbox_transform
from bbox_transform import keep_inside
from generate_anchors import generate_anchors


class AnchorTargetLayer(object):
    """Assign anchors to ground-truth targets.

    It produces:
        1. anchor classification labels
        2. bounding-box regression targets.

    Args:
        rpn_batchsize (int): Number of regions to produce.
        rpn_negative_overlap (float): Anchors with overlap below this
            threshold will be assigned as negative.
        rpn_positive_overlap (float): Anchors with overlap above this
            threshold will be assigned as positive.
        rpn_fg_fraction (float): Fraction of positive regions in the
            set of all regions produced.
        rpn_bbox_inside_weights (tuple of four floats): Four coefficients
            used to calculate bbox_inside_weights.

    """ 

    def __init__(self,
                 rpn_batchsize=128,
                 rpn_negative_overlap=0.3, rpn_positive_overlap=0.7,
                 rpn_fg_fraction=0.5,
                 rpn_bbox_inside_weights=(1., 1., 1., 1.)):
        self.rpn_batchsize = rpn_batchsize
        self.rpn_negative_overlap = rpn_negative_overlap
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_fg_fraction = rpn_fg_fraction
        self.rpn_bbox_inside_weights = rpn_bbox_inside_weights

    def __call__(self, gt_boxes, anchors, feature_size, img_size):
        """Calc targets of classification labels and bbox regression.
        """
        assert gt_boxes.ndim == 3
        assert gt_boxes.shape[0] == 1
        gt_boxes = gt_boxes[0]

        width, height = feature_size
        img_W, img_H = img_size

        n_anchors = len(anchors)
        inds_inside, anchors = keep_inside(anchors, img_H, img_W)
        argmax_overlaps, labels = self._create_labels(
            inds_inside, anchors, gt_boxes)

        # compute bounding box regression targets
        bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

        # calculate inside and outside weights weights
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(
            self.rpn_bbox_inside_weights)
        bbox_outside_weights = self._calc_outside_weights(inds_inside, labels)

        # map up to original set of anchors
        labels = _unmap(labels, n_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(
            bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(
            bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(
            bbox_outside_weights, total_anchors, inds_inside, fill=0)

        # reshape
        labels = labels.reshape(
            (1, height, width, self.n_anchors)).transpose(0, 3, 1, 2)
        labels = labels.astype(np.int32)
        bbox_targets = bbox_targets.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)
        bbox_inside_weights = bbox_inside_weights.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)
        bbox_outside_weights = bbox_outside_weights.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)
        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def _create_labels(self, inds_inside, anchors, gt_boxes):
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        argmax_overlaps, max_overlaps, gt_max_overlaps, gt_argmax_overlaps = \
            self._calc_overlaps(anchors, gt_boxes, inds_inside)

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < self.rpn_negative_overlap] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.rpn_positive_overlap] = 1

        # subsample positive labels if we have too many
        num_fg = int(self.rpn_fg_fraction * self.rpn_batchsize)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.rpn_batchsize - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        return argmax_overlaps, labels

    def _calc_overlaps(self, anchors, gt_boxes, inds_inside):
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        return argmax_overlaps, max_overlaps, gt_max_overlaps, \
            gt_argmax_overlaps

    def _calc_outside_weights(self, inds_inside, labels):
        bbox_outside_weights = np.zeros(
            (len(inds_inside), 4), dtype=np.float32)
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)

        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples

        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        return bbox_outside_weights


def _unmap(self, data, count, inds, fill=0):
    """
        Unmap a subset of item (data) back to the original set of items (of
        size count)
    """

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
