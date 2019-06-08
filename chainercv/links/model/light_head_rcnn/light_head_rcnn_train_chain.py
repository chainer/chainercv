from __future__ import division

import numpy as np
import warnings

import chainer
from chainer.backends import cuda
import chainer.functions as F

from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator
from chainercv.links.model.faster_rcnn.utils.proposal_target_creator \
    import ProposalTargetCreator


class LightHeadRCNNTrainChain(chainer.Chain):

    """Calculate losses for Light Head R-CNN and report them.

    This is used to train Light Head R-CNN in the joint training scheme
    [#LHRCNN]_.

    .. [#LHRCNN] Zeming Li, Chao Peng, Gang Yu, Xiangyu Zhang, Yangdong Deng, \
    Jian Sun. Light-Head R-CNN: In Defense of Two-Stage Object Detector. \
    arXiv preprint arXiv:1711.07264.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.

    Args:
        light_head_rcnn (~light_head_rcnn.links.light_head_rcnn.LightHeadRCNN):
            A Light Head R-CNN model that is going to be trained.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#LHRCNN]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#LHRCNN]_.
        anchor_target_creator: An instantiation of
            :class:`~chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator: An instantiation of
            :class:`~light_head_rcnn.links.model.utils.ProposalTargetCreator`.

    """

    def __init__(
            self, light_head_rcnn,
            rpn_sigma=3., roi_sigma=1., n_ohem_sample=256,
            anchor_target_creator=None, proposal_target_creator=None,
    ):
        super(LightHeadRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.light_head_rcnn = light_head_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.n_ohem_sample = n_ohem_sample

        if anchor_target_creator is None:
            self.anchor_target_creator = AnchorTargetCreator()
        else:
            self.anchor_target_creator = anchor_target_creator

        if proposal_target_creator is None:
            self.proposal_target_creator = ProposalTargetCreator(n_sample=None)
        else:
            self.proposal_target_creator = proposal_target_creator

        self.loc_normalize_mean = light_head_rcnn.loc_normalize_mean
        self.loc_normalize_std = light_head_rcnn.loc_normalize_std

    def __call__(self, imgs, bboxes, labels, scales):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            bboxes (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.array
        if isinstance(labels, chainer.Variable):
            labels = labels.array
        if isinstance(scales, chainer.Variable):
            scales = scales.array
        scales = cuda.to_cpu(scales)

        batch_size, _, H, W = imgs.shape
        img_size = (H, W)

        rpn_features, roi_features = self.light_head_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.light_head_rcnn.rpn(rpn_features, img_size, scales)
        rpn_locs = F.concat(rpn_locs, axis=0)
        rpn_scores = F.concat(rpn_scores, axis=0)

        gt_rpn_locs = []
        gt_rpn_labels = []
        for bbox in bboxes:
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                bbox, anchor, img_size)
            if cuda.get_array_module(rpn_locs.array) != np:
                gt_rpn_loc = cuda.to_gpu(gt_rpn_loc)
                gt_rpn_label = cuda.to_gpu(gt_rpn_label)
            gt_rpn_locs.append(gt_rpn_loc)
            gt_rpn_labels.append(gt_rpn_label)
            del gt_rpn_loc, gt_rpn_label
        gt_rpn_locs = self.xp.concatenate(gt_rpn_locs, axis=0)
        gt_rpn_labels = self.xp.concatenate(gt_rpn_labels, axis=0)

        batch_indices = range(batch_size)
        sample_rois = []
        sample_roi_indices = []
        gt_roi_locs = []
        gt_roi_labels = []

        for batch_index, bbox, label in \
                zip(batch_indices, bboxes, labels):
            roi = rois[roi_indices == batch_index]
            sample_roi, gt_roi_loc, gt_roi_label = \
                self.proposal_target_creator(
                    roi, bbox, label,
                    self.loc_normalize_mean, self.loc_normalize_std)
            del roi
            sample_roi_index = self.xp.full(
                (len(sample_roi),), batch_index, dtype=np.int32)
            sample_rois.append(sample_roi)
            sample_roi_indices.append(sample_roi_index)
            del sample_roi, sample_roi_index
            gt_roi_locs.append(gt_roi_loc)
            gt_roi_labels.append(gt_roi_label)
            del gt_roi_loc, gt_roi_label
        sample_rois = self.xp.concatenate(sample_rois, axis=0)
        sample_roi_indices = self.xp.concatenate(sample_roi_indices, axis=0)
        gt_roi_locs = self.xp.concatenate(gt_roi_locs, axis=0)
        gt_roi_labels = self.xp.concatenate(gt_roi_labels, axis=0)

        roi_cls_locs, roi_scores = self.light_head_rcnn.head(
            roi_features, sample_rois, sample_roi_indices)

        # RPN losses
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs, gt_rpn_locs, gt_rpn_labels, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_scores, gt_rpn_labels)

        # Losses for outputs of the head.
        roi_loc_loss, roi_cls_loss = _ohem_loss(
            roi_cls_locs, roi_scores, gt_roi_locs, gt_roi_labels,
            self.n_ohem_sample * batch_size, self.roi_sigma)
        roi_loc_loss = 2 * roi_loc_loss

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
        chainer.reporter.report(
            {'rpn_loc_loss': rpn_loc_loss,
             'rpn_cls_loss': rpn_cls_loss,
             'roi_loc_loss': roi_loc_loss,
             'roi_cls_loss': roi_cls_loss,
             'loss': loss},
            self)
        return loss


def _ohem_loss(
        roi_cls_locs, roi_scores, gt_roi_locs, gt_roi_labels,
        n_ohem_sample, roi_sigma=1.0
):
    xp = cuda.get_array_module(roi_cls_locs)
    n_sample = roi_cls_locs.shape[0]
    roi_cls_locs = roi_cls_locs.reshape((n_sample, -1, 4))
    roi_locs = roi_cls_locs[xp.arange(n_sample), gt_roi_labels]
    roi_loc_loss = _fast_rcnn_loc_loss(
        roi_locs, gt_roi_locs, gt_roi_labels, roi_sigma, reduce='no')
    roi_cls_loss = F.softmax_cross_entropy(
        roi_scores, gt_roi_labels, reduce='no')
    assert roi_loc_loss.shape == roi_cls_loss.shape

    n_ohem_sample = min(n_ohem_sample, n_sample)
    # sort in CPU because of GPU memory
    roi_cls_loc_loss = cuda.to_cpu(roi_loc_loss.array + roi_cls_loss.array)
    indices = roi_cls_loc_loss.argsort(axis=0)[::-1]
    # filter nan
    indices = np.array(
        [i for i in indices if not np.isnan(roi_cls_loc_loss[i])],
        dtype=np.int32)
    indices = indices[:n_ohem_sample]
    if cuda.get_array_module(roi_loc_loss.array) != np:
        indices = cuda.to_gpu(indices)
    if len(indices) > 0:
        roi_loc_loss = F.sum(roi_loc_loss[indices]) / len(indices)
        roi_cls_loss = F.sum(roi_cls_loss[indices]) / len(indices)
    else:
        roi_loc_loss = chainer.Variable(xp.array(0.0, dtype=xp.float32))
        roi_cls_loss = chainer.Variable(xp.array(0.0, dtype=xp.float32))
        roi_loc_loss.zerograd()
        roi_cls_loss.zerograd()

    return roi_loc_loss, roi_cls_loss


def _smooth_l1_loss_base(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.array < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return F.sum(y, axis=1)


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma, reduce='mean'):
    xp = cuda.get_array_module(pred_loc)

    in_weight = xp.zeros_like(gt_loc)
    # Localization loss is calculated only for positive rois.
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss_base(pred_loc, gt_loc, in_weight, sigma)
    # Normalize by total number of negtive and positive rois.
    if reduce == 'mean':
        loc_loss = F.sum(loc_loss) / xp.sum(gt_label >= 0)
    elif reduce != 'no':
        warnings.warn('no reduce option: {}'.format(reduce))
    return loc_loss
