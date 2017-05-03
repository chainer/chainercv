# Mofidied by:
# Copyright (c) 2016 Shunta Saito

# Original work by:
# -----------------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# https://github.com/rbgirshick/py-faster-rcnn
# -----------------------------------------------------------------------------

from chainer.cuda import to_cpu
import numpy as np

from bbox_transform import bbox_transform_inv
from bbox_transform import clip_boxes
from generate_anchors import generate_anchors

from nms_cpu import nms_cpu
from nms_gpu import nms_gpu


class ProposalLayer(object):
    """Generate deterministic proposal regions (All on CPU)
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, use_gpu_nms=True,
                 rpn_nms_thresh=0.7,
                 train_rpn_pre_nms_top_n=12000,
                 train_rpn_post_nms_top_n=2000,
                 test_rpn_pre_nms_top_n=6000,
                 test_rpn_post_nms_top_n=300,
                 rpn_min_size=16):
        self.use_gpu_nms = use_gpu_nms
        self.rpn_nms_thresh = rpn_nms_thresh
        self.train_rpn_pre_nms_top_n = train_rpn_pre_nms_top_n
        self.train_rpn_post_nms_top_n = train_rpn_post_nms_top_n
        self.test_rpn_pre_nms_top_n = test_rpn_pre_nms_top_n
        self.test_rpn_post_nms_top_n = test_rpn_pre_nms_top_n 
        self.rpn_min_size = rpn_min_size

    def __call__(self, rpn_cls_prob, rpn_bbox_pred,
                 anchors, img_size, scale=1., train=False):
        """
        Args:
            rpn_cls_prob:
            rpn_bbox_pred:
        """
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        pre_nms_topN = self.train_rpn_pre_nms_top_n \
            if train else self.test_rpn_pre_nms_top_n
        post_nms_topN = self.train_rpn_post_nms_top_n \
            if train else self.test_rpn_post_nms_top_n

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        n_anchors = rpn_cls_prob.shape[1] / 2
        scores = to_cpu(rpn_cls_prob.data[:, n_anchors:, :, :])
        bbox_deltas = to_cpu(rpn_bbox_pred.data)

        # Transpose and reshape predicted bbox transformations and scores
        # to get them into the same order as the anchors:
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors 
        # into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, img_size)

        # 3. remove predicted boxes with either height or width < threshold
        keep = _filter_boxes(proposals, self.rpn_min_size * scale)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if self.use_gpu_nms:
            keep = nms_gpu(np.hstack((proposals, scores)), self.rpn_nms_thresh)
        else:
            keep = nms_cpu(np.hstack((proposals, scores)), self.rpn_nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        rois = np.hstack((batch_inds, proposals)).astype(np.float32, copy=False)

        return rois


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
