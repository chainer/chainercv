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
from generate_anchors import generate_anchors


class AnchorTargetLayer(object):
    """Assign anchors to ground-truth targets
    Produces anchor classification labels and bounding-box regression targets.
    """

    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_CLOBBER_POSITIVES = False
    RPN_FG_FRACTION = 0.5
    RPN_BATCHSIZE = 256
    RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    RPN_POSITIVE_WEIGHT = -1.0

    def __init__(self, feat_stride=16, scales=2 ** np.arange(3, 6)):
        self.feat_stride = feat_stride
        self.anchors = generate_anchors(scales=scales)
        self.n_anchors = self.anchors.shape[0]
        self.allowed_border = 0

    def __call__(self, gt_boxes, feature_shape, img_shape):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap
        # gt_boxes (x1, y1, x2, y2, label)
        # im_info: (height, width, scale)
        # assert x.data.shape[0] == 1, \
        #    'Only single item batches are supported'

        # map of shape (..., H, W)
        # height, width = x.data.shape[-2:]
        assert gt_boxes.ndim == 3
        assert gt_boxes.shape[0] == 1
        gt_boxes = gt_boxes[0]

        height, width = feature_shape
        img_H, img_W = img_shape

        shifts = self._generate_shifts(width, height)
        all_anchors, total_anchors = self._generate_proposals(shifts)
        inds_inside, anchors = self._keep_inside(all_anchors, img_H, img_W)
        argmax_overlaps, labels = self._create_labels(
            inds_inside, anchors, gt_boxes)

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = self._compute_targets(
            anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = self._calc_inside_weights(inds_inside, labels)
        bbox_outside_weights = self._calc_outside_weights(inds_inside, labels)

        # map up to original set of anchors
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            self._mapup_to_anchors(
                labels, total_anchors, inds_inside, bbox_targets,
                bbox_inside_weights, bbox_outside_weights)

        # labels
        labels = labels.reshape(
            (1, height, width, self.n_anchors)).transpose(0, 3, 1, 2)
        labels = labels.astype(np.int32)

        # bbox_targets
        bbox_targets = bbox_targets.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def _generate_shifts(self, width, height):
        """
        1. Generate proposals from bbox deltas and shifted anchors
        width and height mean the spatial dimensions of feat map

        Returns:
            an array of shape (width * height, 4)
        """

        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        return shifts

    def _keep_inside(self, all_anchors, img_H, img_W):
        """
        im_info's height and width information is being used
        """
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self.allowed_border) &
            (all_anchors[:, 1] >= -self.allowed_border) &
            (all_anchors[:, 2] < img_W + self.allowed_border) &  # width
            (all_anchors[:, 3] < img_H + self.allowed_border)    # height
        )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        return inds_inside, anchors

    def _calc_inside_weights(self, inds_inside, labels):
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(
            self.RPN_BBOX_INSIDE_WEIGHTS)

        return bbox_inside_weights

    def _generate_proposals(self, shifts):
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K * A, 4) shifted anchorsd
        A = self.n_anchors
        K = shifts.shape[0]
        all_anchors = (self.anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        return all_anchors, total_anchors

    def _create_labels(self, inds_inside, anchors, gt_boxes):
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        argmax_overlaps, max_overlaps, gt_max_overlaps, gt_argmax_overlaps = \
            self._calc_overlaps(anchors, gt_boxes, inds_inside)

        if not self.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.RPN_POSITIVE_OVERLAP] = 1

        if self.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber
            # positives
            labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(self.RPN_FG_FRACTION * self.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            # print "was %s inds, disabling %s, now %s inds" % (
            # len(bg_inds), len(disable_inds), np.sum(labels == 0))

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
        if self.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((self.RPN_POSITIVE_WEIGHT > 0) &
                    (self.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (self.RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
            negative_weights = ((1.0 - self.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        return bbox_outside_weights

    def _mapup_to_anchors(
            self, labels, total_anchors, inds_inside, bbox_targets,
            bbox_inside_weights, bbox_outside_weights):
        # map up to original set of anchors
        labels = self._unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = self._unmap(
            bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = self._unmap(
            bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = self._unmap(
            bbox_outside_weights, total_anchors, inds_inside, fill=0)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

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

    def _compute_targets(self, ex_rois, gt_rois):
        """
            Compute bounding-box regression targets for an image.
        """

        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 5

        return bbox_transform(ex_rois, gt_rois[:, :4]).astype(
            np.float32, copy=False)
