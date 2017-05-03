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

from chainer import cuda
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

    def __call__(self, rpn_bbox_pred, rpn_cls_prob,
                 anchor, img_size, scale=1., train=False):
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
        # return the top proposals (-> RoIs top, score top)
        xp = cuda.get_array_module(rpn_cls_prob)

        pre_nms_topN = self.train_rpn_pre_nms_top_n \
            if train else self.test_rpn_pre_nms_top_n
        post_nms_topN = self.train_rpn_post_nms_top_n \
            if train else self.test_rpn_post_nms_top_n

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        n_anchor = rpn_cls_prob.shape[1] / 2
        score = rpn_cls_prob.data[:, n_anchor:, :, :]
        bbox_deltas = rpn_bbox_pred.data

        # Transpose and reshape predicted bbox transformations and score
        # to get them into the same order as the anchors:
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        score = score.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors
        # into proposal via bbox transformations
        proposal = bbox_transform_inv(anchor, bbox_deltas)

        # 2. clip predicted boxes to image
        proposal = clip_boxes(proposal, img_size)

        # 3. remove predicted boxes with either height or width < threshold
        min_size = self.rpn_min_size * scale
        ws = proposal[:, 2] - proposal[:, 0] + 1
        hs = proposal[:, 3] - proposal[:, 1] + 1
        keep = xp.where((ws >= min_size) & (hs >= min_size))[0]
        proposal = proposal[keep, :]
        score = score[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        score = cuda.to_cpu(score)
        proposal = cuda.to_cpu(proposal)
        order = score.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposal = proposal[order, :]
        score = score[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposal (-> RoIs top)
        if self.use_gpu_nms:
            keep = nms_gpu(np.hstack((proposal, score)), self.rpn_nms_thresh)
        else:
            keep = nms_cpu(np.hstack((proposal, score)), self.rpn_nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposal = proposal[keep, :]
        score = score[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        if xp != np:
            proposal = cuda.to_gpu(proposal)
        batch_inds = xp.zeros((proposal.shape[0], 1), dtype=np.float32)
        roi = xp.hstack((batch_inds, proposal)).astype(np.float32, copy=False)

        return roi
