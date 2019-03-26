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


class FCISTrainChain(chainer.Chain):

    """Calculate losses for FCIS and report them.

    This is used to train FCIS in the joint training scheme [#FCISCVPR]_.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`roi_mask_loss`: The mask loss for the head module.

    .. [#FCISCVPR] Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei. \
    Fully Convolutional Instance-aware Semantic Segmentation. CVPR 2017.

    Args:
        fcis (~chainercv.experimental.links.model.fcis.FCIS):
            A FCIS model for training.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#FCISCVPR]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#FCISCVPR]_.
        anchor_target_creator: An instantiation of
            :class:`~chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator: An instantiation of
            :class:`~chainercv.experimental.links.model.fcis.ProposalTargetCreator`.

    """

    def __init__(
            self, fcis,
            rpn_sigma=3.0, roi_sigma=1.0,
            anchor_target_creator=AnchorTargetCreator(),
            proposal_target_creator=ProposalTargetCreator()
    ):

        super(FCISTrainChain, self).__init__()
        with self.init_scope():
            self.fcis = fcis
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.mask_size = self.fcis.head.roi_size

        self.loc_normalize_mean = fcis.loc_normalize_mean
        self.loc_normalize_std = fcis.loc_normalize_std

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

    def forward(self, imgs, masks, labels, bboxes, scale):
        """Forward FCIS and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.
        * :math:`H` is the image height.
        * :math:`W` is the image width.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            masks (~chainer.Variable): A batch of masks.
                Its shape is :math:`(N, R, H, W)`.
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            bboxes (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            scale (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        if isinstance(masks, chainer.Variable):
            masks = masks.array
        if isinstance(labels, chainer.Variable):
            labels = labels.array
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.array
        if isinstance(scale, chainer.Variable):
            scale = scale.array
        scale = scale.item()

        n = masks.shape[0]
        # batch size = 1
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)
        assert img_size == masks.shape[2:]

        rpn_features, roi_features = self.fcis.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.fcis.rpn(
            rpn_features, img_size, scale)

        # batch size = 1
        mask = masks[0]
        label = labels[0]
        bbox = bboxes[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        sample_roi, gt_roi_mask, gt_roi_label, gt_roi_loc = \
            self.proposal_target_creator(
                roi, mask, label, bbox, self.loc_normalize_mean,
                self.loc_normalize_std, self.mask_size)

        sample_roi_index = self.xp.zeros(
            (len(sample_roi),), dtype=np.int32)
        roi_ag_seg_score, roi_ag_loc, roi_cls_score, _, _ = self.fcis.head(
            roi_features, sample_roi, sample_roi_index, img_size,
            gt_roi_label, iter2=False)

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
        n_roi = roi_ag_loc.shape[0]
        gt_roi_fg_label = (gt_roi_label > 0).astype(np.int)
        roi_loc = roi_ag_loc[self.xp.arange(n_roi), gt_roi_fg_label]

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
