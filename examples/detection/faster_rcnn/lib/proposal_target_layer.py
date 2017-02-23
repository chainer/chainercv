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
    """Assign object detection proposals to ground-truth targets
    Produces proposal classification labels and bounding-box regression
    targets.
    """

    BATCH_SIZE = 128  # number of regions of interest [ROIs]
    # Fraction of minibatch that is labeled foreground (i.e. class > 0)
    FG_FRACTION = 0.25
    # Deprecated (inside weights)
    BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    # Normalize the targets using "precomputed" (or made up) means and stdevs
    # (BBOX_NORMALIZE_TARGETS must also be True)
    BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
    # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    FG_THRESH = 0.5
    # Overlap threshold for a ROI to be considered background (class = 0 if
    # overlap in [LO, HI))
    BG_THRESH_HI = 0.5
    BG_THRESH_LO = 0.1

    def __init__(self, num_classes=21):
        self._num_classes = num_classes

    def __call__(self, all_rois, gt_boxes):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        # Include ground-truth boxes in the set of candidate rois
        assert gt_boxes.shape[0] == 1
        gt_boxes = gt_boxes[0]
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

        num_images = 1
        rois_per_image = self.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(self.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)
        labels = labels.astype(np.int32)
        rois = rois.astype(np.float32)

        return rois, labels, bbox_targets, bbox_inside_weights, \
            np.array(bbox_inside_weights > 0).astype(np.float32)

    def _get_bbox_regression_labels(self, bbox_target_data, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation
        used by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """

        clss = bbox_target_data[:, 0]
        bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(clss > 0)[0]
        for ind in inds:
            cls = int(clss[ind])
            start = int(4 * cls)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = self.BBOX_INSIDE_WEIGHTS
        return bbox_targets, bbox_inside_weights

    def _compute_targets(self, ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)
        if self.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(self.BBOX_NORMALIZE_MEANS)
                        ) / np.array(self.BBOX_NORMALIZE_STDS))
        return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

    def _sample_rois(
            self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image,
            num_classes):
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
        fg_inds = np.where(max_overlaps >= self.FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < self.BG_THRESH_HI) &
                           (max_overlaps >= self.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
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
            self._get_bbox_regression_labels(bbox_target_data, num_classes)

        return labels, rois, bbox_targets, bbox_inside_weights
