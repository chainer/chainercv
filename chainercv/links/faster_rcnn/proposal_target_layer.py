# Modified by:
# Copyright (c) 2016 Shunta Saito

# Original codes by:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


import numpy as np
import numpy.random as npr
from bbox_transform import bbox_transform

from bbox import bbox_overlaps
from bbox_transform import get_bbox_regression_label

import chainer
from chainer import cuda


class ProposalTargetLayer(object):
    """Assign proposals to ground-truth targets.

    Generates training targets/labels for each object proposal: classification
    labels 0 - K (bg or object class 1, ... , K) and bbox regression targets in
    that case that the label is > 0.

    It produces:
        1. classification labels for each proposal
        2. proposed bounding-box regression targets.

    Args:
        n_class (int): Number of classes to categorize.
        batch_size (int): Number of regions to produce.
        bbox_normalize_target_precomputed (bool): Normalize the targets
            using :obj:`bbox_normalize_means` and :obj:`bbox_normalize_stds`.
        bbox_normalize_mean (tuple of four floats): Mean values to normalize
            coordinates of bouding boxes.
        bbox_normalize_std (tupler of four floats): Standard deviation of
            the coordinates of bounding boxes.
        bbox_inside_weight (tuple of four floats):
        fg_fraction (float): Fraction of regions that is labeled foreground.
        fg_thresh (float): Overlap threshold for a ROI to be considered
            foreground.
        bg_thresh_hi (float): ROI is considered to be background if overlap is
            in [:obj:`bg_thresh_hi`, :obj:`bg_thresh_hi`).
            bg_thresh_lo (float): See :obj:`bg_thresh_hi`.

    """

    def __init__(self, n_class=21,
                 batch_size=128,
                 bbox_normalize_target_precomputed=True,
                 bbox_normalize_mean=(0., 0., 0., 0.),
                 bbox_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 bbox_inside_weight=(1., 1., 1., 1.),
                 fg_fraction=0.25,
                 fg_thresh=0.5, bg_thresh_hi=0.5, bg_thresh_lo=0.0
                 ):
        self.n_class = n_class
        self.batch_size = batch_size
        self.fg_fraction = fg_fraction
        self.bbox_inside_weight = bbox_inside_weight
        self.bbox_normalize_target_precomputed =\
            bbox_normalize_target_precomputed
        self.bbox_normalize_mean = bbox_normalize_mean
        self.bbox_normalize_std = bbox_normalize_std
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo

    def __call__(self, roi, bbox, label):
        """It assigns labels to proposals from RPN.

        Args:
            proposals (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                :math:`(n_proposals, 4)`-shaped array. These proposals come
                from RegionProposalNetwork.
            gt_boxes (:class:`~chainer.Variable`):
                A :math:`(1, n_gt_boxes, 5)`-shaped array, each of which is a
                4-dimensional vector that represents
                :math:`(x1, y1, x2, y2, cls_id)` of each ground truth bbox.
                The scale of them are at the input image scale.

        Returns:
            roi_sample (~ndarray)
            label_sample (~ndarray)
            bbox_target_sample (~ndarray)
            bbox_inside_weight (~ndarray)
            bbox_outside_weight (~ndarray)

        """
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        # Include ground-truth boxes in the set of candidate rois
        if isinstance(bbox, chainer.Variable):
            bbox = bbox.data
        if isinstance(label, chainer.Variable):
            label = label.data
        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        bbox = cuda.to_cpu(bbox)
        label = cuda.to_cpu(label)

        assert bbox.ndim == 3
        n_image, n_bbox, _ = bbox.shape
        assert bbox.shape[0] == 1
        assert label.shape[0] == 1
        # Sanity check: single batch only
        assert np.all(roi[:, 0] == 0), \
            'Only single item batches are supported'

        bbox = bbox[0]
        label = label[0]

        gt_roi = np.hstack((np.zeros((n_bbox, 1), dtype=bbox.dtype), bbox))
        roi = np.vstack((roi, gt_roi))

        rois_per_image = self.batch_size / n_image
        fg_rois_per_image = np.round(self.fg_fraction * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        label_sample, roi_sample, bbox_target_sample, bbox_inside_weight =\
            self._sample_roi(
                roi, bbox, label, fg_rois_per_image,
                rois_per_image, self.n_class)
        label_sample = label_sample.astype(np.int32)
        roi_sample = roi_sample.astype(np.float32)

        bbox_outside_weight = (bbox_inside_weight > 0).astype(np.float32)

        if xp != np:
            roi_sample = cuda.to_gpu(roi_sample)
            bbox_target_sample = cuda.to_gpu(bbox_target_sample)
            label_sample = cuda.to_gpu(label_sample)
            bbox_inside_weight = cuda.to_gpu(bbox_inside_weight)
            bbox_outside_weight = cuda.to_gpu(bbox_outside_weight)
        return roi_sample, bbox_target_sample, label_sample,\
            bbox_inside_weight, bbox_outside_weight

    def _sample_roi(
            self, roi, bbox, label, fg_rois_per_image, rois_per_image,
            n_class):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(roi[:, 1:5], dtype=np.float),
            np.ascontiguousarray(bbox, dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        label_sample = label[gt_assignment]

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= self.fg_thresh)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < self.bg_thresh_hi) &
                           (max_overlaps >= self.bg_thresh_lo))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        # Select sampled values from various arrays:
        label_sample = label_sample[keep_inds]
        # Clamp labels for the background RoIs to 0
        label_sample[fg_rois_per_this_image:] = 0
        roi_sample = roi[keep_inds]

        bbox_sample = bbox_transform(
            roi_sample[:, 1:5], bbox[gt_assignment[keep_inds]])
        if self.bbox_normalize_target_precomputed:
            # Optionally normalize targets by a precomputed mean and stdev
            bbox_sample = ((bbox_sample - np.array(self.bbox_normalize_mean)
                            ) / np.array(self.bbox_normalize_std))

        bbox_target_sample, bbox_inside_weight = \
            get_bbox_regression_label(
                bbox_sample, label_sample, n_class, self.bbox_inside_weight)
        return label_sample, roi_sample, bbox_target_sample, bbox_inside_weight
