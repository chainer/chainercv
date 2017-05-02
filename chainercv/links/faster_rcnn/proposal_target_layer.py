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
        bbox_normalize_targets_precomputed (bool): Normalize the targets
            using :obj:`bbox_normalize_means` and :obj:`bbox_normalize_stds`.
        bbox_normalize_means (tuple of four floats): Mean values to normalize
            coordinates of bouding boxes.
        bbox_normalize_stds (tupler of four floats): Standard deviation of
            the coordinates of bounding boxes.
        bbox_inside_weights (tuple of four floats):
        fg_fraction (float): Fraction of regions that is labeled foreground.
        fg_thresh (float): Overlap threshold for a ROI to be considered
            foreground.
        bg_thresh_hi (float): ROI is considered to be background if overlap is
            in [:obj:`bg_thresh_hi`, :obj:`bg_thresh_hi`).
            bg_thresh_lo (float): See :obj:`bg_thresh_hi`.

    """

    def __init__(self, n_class=21,
                 batch_size=256,
                 bbox_normalize_targets_precomputed=True,
                 bbox_normalize_means=(0., 0., 0., 0.),
                 bbox_normalize_stds=(0.1, 0.1, 0.2, 0.2),
                 bbox_inside_weights=(1., 1., 1., 1.),
                 fg_fraction=0.25,
                 fg_thresh=0.5, bg_thresh_hi=0.5, bg_thresh_lo=0.0
                 ):
        self.n_class = n_class
        self.batch_size = batch_size
        self.fg_fraction = fg_fraction
        self.bbox_inside_weights = bbox_inside_weights
        self.bbox_normalize_targets_precomputed =\
            bbox_normalize_targets_precomputed
        self.bbox_normalize_means = bbox_normalized_means
        self.bbox_normalize_stds = bbox_normalize_stds
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo

    def __call__(self, all_rois, gt_boxes):
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

        """
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        # Include ground-truth boxes in the set of candidate rois
        assert gt_boxes.ndim == 3
        n_image = gt_boxes.shape[0]
        assert gt_boxes.shape[0] == 1
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

        gt_boxes = gt_boxes[0]
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))

        rois_per_image = self.batch_size / n_image
        fg_rois_per_image = np.round(self.fg_fraction * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self.n_class)
        labels = labels.astype(np.int32)
        rois = rois.astype(np.float32)

        return rois, labels, bbox_targets, bbox_inside_weights, \
            np.array(bbox_inside_weights > 0).astype(np.float32)

    def _get_bbox_regression_labels(self, bbox_target_data, n_class):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation
        used by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """

        clss = bbox_target_data[:, 0]
        bbox_targets = np.zeros((clss.size, 4 * n_class), dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(clss > 0)[0]
        for ind in inds:
            cls = int(clss[ind])
            start = int(4 * cls)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = self.bbox_inside_weights
        return bbox_targets, bbox_inside_weights

    def _compute_targets(self, ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)
        if self.bbox_normalize_targets_precomputed:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(self.bbox_normalized_means)
                        ) / np.array(self.bbox_normalize_stds))
        return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

    def _sample_rois(
            self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image,
            n_class):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]


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
        labels = labels[keep_inds]
        # Clamp labels for the background RoIs to 0
        labels[fg_rois_per_this_image:] = 0
        rois = all_rois[keep_inds]

        bbox_target_data = self._compute_targets(
            rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels(bbox_target_data, n_class)

        return labels, rois, bbox_targets, bbox_inside_weights
