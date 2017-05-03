import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from anchor_target_layer import AnchorTargetLayer
from generate_anchors import generate_anchors
from proposal_layer import ProposalLayer

from chainercv.functions.smooth_l1_loss import smooth_l1_loss


class RegionProposalNetwork(chainer.Chain):

    def __init__(
            self, in_ch=512, out_ch=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16, rpn_sigma=3.0,
            anchor_target_layer_params={},
            proposal_layer_params={},
    ):
        self.anchor_base = generate_anchors(
            scales=np.array(anchor_scales), ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalLayer(**proposal_layer_params)

        n_anchor = self.anchor_base.shape[0]
        initializer = chainer.initializers.Normal(scale=0.01)
        super(RegionProposalNetwork, self).__init__(
            rpn_conv_3x3=L.Convolution2D(
                in_ch, out_ch, 3, 1, 1, initialW=initializer),
            rpn_cls_score=L.Convolution2D(
                out_ch, 2 * n_anchor, 1, 1, 0, initialW=initializer),
            rpn_bbox_pred=L.Convolution2D(
                out_ch, 4 * n_anchor, 1, 1, 0, initialW=initializer)
        )

    def __call__(self, x, img_size, scale=1., train=False):
        """aaa

        x:  (N, C, H, W)
        img_shape (img_W, img_H)

        Returns:
            rpn_bbox_pred  (~chainer.Variable)
            rpn_cls_score  (~chainer.Variable)
            roi  (~ndarray)
            anchor  (~ndarray)

        """
        xp = cuda.get_array_module(x)
        n = x.data.shape[0]
        assert n == 1
        h = F.relu(self.rpn_conv_3x3(x))
        rpn_cls_score = self.rpn_cls_score(h)  # (N, 2 * A, H/16, W/16)
        c, hh, ww = rpn_cls_score.shape[1:]
        # take probability against (N, 2, A * H/16 * W/16)
        rpn_cls_prob = F.softmax(rpn_cls_score.reshape(n, 2, -1))
        rpn_cls_prob = rpn_cls_prob.reshape(n, c, hh, ww)
        rpn_bbox_pred = self.rpn_bbox_pred(h)

        # enumerate all shifted anchors
        anchor = _enumerate_shifted_anchor(
            xp.array(self.anchor_base), self.feat_stride, ww, hh)
        roi = self.proposal_layer(
            rpn_cls_prob, rpn_bbox_pred, anchor, img_size,
            scale=scale, train=train)
        return rpn_bbox_pred, rpn_cls_score, roi, anchor


class RegionProposalNetworkLoss(object):

    """A loss function that produces loss given inputs

    """

    def __init__(self, rpn_sigma=3., anchor_target_layer_params={}):
        self.anchor_target_layer = AnchorTargetLayer(
            **anchor_target_layer_params)
        self.rpn_sigma = rpn_sigma

    def __call__(self, rpn_bbox_pred, rpn_cls_score,
                 bbox, anchor, img_size):
        """
        Args:
            rpn_bbox_pred (~chainer.Variable)
            rpn_cls_score (~chainer.Variable)
            bbox (~ndarray or chainer.Variable)
            anchor (~ndarray)
            img_size (tuple of ints)
        """
        hh, ww = rpn_bbox_pred.shape[2:]
        n = bbox.shape[0]
        assert n == 1

        rpn_label, rpn_bbox_target, rpn_bbox_inside_weight, \
            rpn_bbox_outside_weight = self.anchor_target_layer(
                bbox, anchor, (ww, hh), img_size)
        rpn_label = rpn_label.reshape((n, -1))

        rpn_cls_score = rpn_cls_score.reshape(1, 2, -1)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_cls_score, rpn_label)
        rpn_loss_bbox = smooth_l1_loss(
            rpn_bbox_pred, rpn_bbox_target, rpn_bbox_inside_weight,
            rpn_bbox_outside_weight, self.rpn_sigma)
        return rpn_loss_bbox, rpn_cls_loss


def _enumerate_shifted_anchor(anchor, feat_stride, width, height):
    xp = cuda.get_array_module(anchor)
    # 1. Generate proposals from bbox deltas and shifted anchors
    # Enumerate all shifts
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchor.shape[0]
    K = shift.shape[0]
    anchor_shifted = anchor.reshape((1, A, 4)) + \
        shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor_shifted = anchor_shifted.reshape((K * A, 4)).astype(np.float32)
    return anchor_shifted
