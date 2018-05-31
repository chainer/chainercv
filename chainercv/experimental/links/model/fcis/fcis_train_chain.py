from __future__ import division

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F

from chainercv.experimental.links.model.fcis.utils.proposal_target_creator \
    import ProposalTargetCreator
from chainercv.links.model.faster_rcnn.faster_rcnn_train_chain \
    import _fast_rcnn_loc_loss
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator \
    import AnchorTargetCreator
from chainercv.utils import mask_to_bbox


class FCISTrainChain(chainer.Chain):

    def __init__(
            self, fcis,
            rpn_sigma=3.0, roi_sigma=1.0,
            n_sample=128,
            pos_ratio=0.25, pos_iou_thresh=0.5,
            neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
            binary_thresh=0.4
    ):

        super(FCISTrainChain, self).__init__()
        with self.init_scope():
            self.fcis = fcis
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.n_sample = n_sample
        self.mask_size = self.fcis.head.roi_size

        self.loc_normalize_mean = fcis.loc_normalize_mean
        self.loc_normalize_std = fcis.loc_normalize_std

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator(
            n_sample=n_sample,
            loc_normalize_mean=self.loc_normalize_mean,
            loc_normalize_std=self.loc_normalize_std,
            pos_ratio=pos_ratio, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh_hi=neg_iou_thresh_hi,
            neg_iou_thresh_lo=neg_iou_thresh_lo,
            mask_size=self.mask_size, binary_thresh=binary_thresh)

    def __call__(self, x, masks, labels, scale=1.0):
        n = masks.shape[0]
        # batch size = 1
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = x.shape
        img_size = (H, W)
        assert img_size == masks.shape[2:]

        rpn_features, roi_features = self.fcis.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.fcis.rpn(
            rpn_features, img_size, scale)

        # batch size = 1
        mask = masks[0]
        bbox = mask_to_bbox(mask)
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_mask, gt_roi_label = \
            self.proposal_target_creator(roi, bbox, mask, label)

        sample_roi_index = self.xp.zeros(
            (len(sample_roi),), dtype=np.int32)
        roi_ag_seg_score, roi_ag_loc, roi_cls_score, _, _ = self.fcis.head(
            roi_features, sample_roi, sample_roi_index, img_size, gt_roi_label)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)

        # CPU -> GPU
        if cuda.get_array_module(rpn_loc.array) != np:
            gt_rpn_loc = cuda.to_gpu(gt_rpn_loc)
            gt_rpn_label = cuda.to_gpu(gt_rpn_label)

        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head
        n_rois = roi_ag_loc.shape[0]
        gt_roi_fg_label = (gt_roi_label > 0).astype(np.int)
        roi_loc = roi_ag_loc[self.xp.arange(n_rois), gt_roi_fg_label]

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_cls_score, gt_roi_label)
        # normalize by every (valid and invalid) instances
        roi_mask_loss = F.softmax_cross_entropy(
            roi_ag_seg_score, gt_roi_mask, normalize=False) \
            * 10.0 / self.mask_size / self.mask_size

        loss = rpn_loc_loss + rpn_cls_loss \
            + roi_loc_loss + roi_cls_loss + roi_mask_loss
        chainer.reporter.report({
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'roi_loc_loss': roi_loc_loss,
            'roi_cls_loss': roi_cls_loss,
            'roi_mask_loss': roi_mask_loss,
            'loss': loss,
        }, self)

        return loss
