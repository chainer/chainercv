import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import \
    AnchorTargetCreator
from chainercv.links.model.faster_rcnn.utils.proposal_target_creator import \
    ProposalTargetCreator


class FasterRCNNTrainChain(chainer.Chain):

    """Calculate Faster R-CNN losses and report them.

    This is used to train Faster R-CNN in the joint training scheme [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        faster_rcnn (~chainercv.links.model.faster_rcnn.FasterRCNN):
            A Faster R-CNN model to train.
        rpn_sigma (float): Sigma parameter for localization loss
            of Region Proposal Network (RPN).
        sigma (float): Sigma paramter for localization loss of
            calculated from the output of the head.
        anchor_target_creator: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator_params: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.ProposalTargetCreator`.

    """

    def __init__(self, faster_rcnn, rpn_sigma=3., sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator()):
        super(FasterRCNNTrainChain, self).__init__(faster_rcnn=faster_rcnn)
        self.rpn_sigma = rpn_sigma
        self.sigma = sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

        self.n_class = faster_rcnn.n_class
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.train = True

    def __call__(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the number of batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently only :math:`N=1` is supported.


        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            bboxes (~chainer.Variable): Batched bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~chainer.Variable): Batched labels.
                Its shape is :math:`(N, R)`.
            scale (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Loss scalar variable.
            This is a sum of losses for Region Proposal Network and
            the head module.

        """
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        scale = np.asscalar(cuda.to_cpu(scale))
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('currently only batch size 1 is supported')

        _, _, H, W = imgs.shape
        img_size = (W, H)

        features = self.faster_rcnn.extractor(imgs, test=not self.train)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
            features, img_size, scale, test=not self.train)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi, bbox, label,
            self.loc_normalize_mean, self.loc_normalize_std)
        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features, sample_roi, sample_roi_index, test=not self.train)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)

        rpn_loc_weight = self.xp.zeros_like(gt_rpn_loc)
        rpn_loc_weight[gt_rpn_label == 1] = 1
        rpn_loc_loss = _smooth_l1_loss(
            rpn_loc, gt_rpn_loc,
            rpn_loc_weight, self.rpn_sigma)
        rpn_loc_loss /= self.xp.sum(gt_rpn_label >= 0)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head.
        gt_roi_cls_loc, roi_loc_in_weight = _loc2cls_loc(
            gt_roi_loc, gt_roi_label, self.n_class)
        loc_loss = _smooth_l1_loss(
            roi_cls_loc, gt_roi_cls_loc,
            roi_loc_in_weight, self.sigma)
        loc_loss /= roi_score.shape[0]
        cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)

        loss = rpn_loc_loss + rpn_cls_loss + loc_loss + cls_loss
        chainer.reporter.report({'rpn_loc_loss': rpn_loc_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'loc_loss': loc_loss,
                                 'cls_loss': cls_loss,
                                 'loss': loss},
                                self)
        return loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))

    return F.sum(y)


def _loc2cls_loc(loc, label, n_class):
    # From loc (R, 4) and label (R,), this function computes
    # cls_loc (R, L * 4).
    # Only one class has non-zero targets in this representation.
    xp = cuda.get_array_module(loc)
    n_bbox = label.shape[0]
    cls_loc = xp.zeros((n_bbox, n_class * 4), dtype=np.float32)
    loc_in_weight = xp.zeros_like(cls_loc)
    index = xp.where(label > 0)[0]
    for ind in index:
        l = int(label[ind])
        start = int(4 * l)
        end = int(start + 4)
        cls_loc[ind, start:end] = loc[ind]
        loc_in_weight[ind, start:end] = 1
    return cls_loc, loc_in_weight
