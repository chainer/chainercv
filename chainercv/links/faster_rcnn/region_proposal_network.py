import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


from chainercv.links import generate_anchor_base
from chainercv.links import ProposalCreator


class RegionProposalNetwork(chainer.Chain):

    """Region Proposal Networks introduced in Faster RCNN.

    This is Region Proposal Networks introduced in Faster RCNN.
    This takes features extracted from an image and predict
    class agnostic bounding boxes around "objects".

    Args:
        n_in (int): Channel size of input.
        n_mid (int): Channel size of the intermediate tensor.
        ratios (list of floats): Anchors with ratios contained in this list
            will be generated. Ratio is the ratio of the height by the width.
        anchor_scales (list of numbers): Values in :obj:`scales` determine area
            of possibly generated anchors. Those areas will be square of an
            element in :obj:`scales` times the original area of the
            reference window.
        feat_stride (int): Stride size after extracting features from an
            image.
        proposal_creator_params (dict): Key valued paramters for
            :obj:`chainercv.links.ProposalCreator`.

    .. seealso::
        :obj:`chainercv.links.ProposalCreator`

    """

    def __init__(
            self, n_in=512, n_mid=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params={},
    ):
        self.anchor_base = generate_anchor_base(
            scales=np.array(anchor_scales), ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(**proposal_creator_params)

        n_anchor = self.anchor_base.shape[0]
        initializer = chainer.initializers.Normal(scale=0.01)
        super(RegionProposalNetwork, self).__init__(
            rpn_conv_3x3=L.Convolution2D(
                n_in, n_mid, 3, 1, 1, initialW=initializer),
            rpn_cls_score=L.Convolution2D(
                n_mid, 2 * n_anchor, 1, 1, 0, initialW=initializer),
            rpn_bbox_pred=L.Convolution2D(
                n_mid, 4 * n_anchor, 1, 1, 0, initialW=initializer)
        )

    def __call__(self, x, img_size, scale=1., train=False):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.
        * :math:`R` is number of rois produced.

        Args:
            x (~chainer.Variable): Feature extracted from an image.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`width, height`,
                which contains image size after scaling if any.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.
            train (bool): If this is in train mode or not.
                Default value is :obj:`False`.

        Returns:
            (~chainer.Variable, ~chainer.Variable, array, array):

            This is a tuple of four following values.

            * **rpn_bbox_pred**: Predicted regression targets for anchors. \
                Its shape is :math:`(1, 4 A, H, W)`.
            * **rpn_cls_prob**:  Predicted foreground probability for \
                anchors. Its shape is :math:`(1, 2 A, H, W)`.
            * **roi**: An array whose shape is :math:`(S, 5)`. The \
                second axis contains \
                :obj:`(batch_index, x_min, y_min, x_max, y_max)` of \
                each region of interests.
            * **anchor**: Coordinates of anchors. Its shape is \
                :math:`(R, 4)`. The second axis contains x and y coordinates \
                of left top vertices and right bottom vertices.

        """
        xp = cuda.get_array_module(x)
        n = x.data.shape[0]
        h = F.relu(self.rpn_conv_3x3(x))
        rpn_cls_score = self.rpn_cls_score(h)
        c, hh, ww = rpn_cls_score.shape[1:]
        rpn_cls_prob = F.softmax(rpn_cls_score.reshape(n, 2, -1))
        rpn_cls_prob = rpn_cls_prob.reshape(n, c, hh, ww)
        rpn_bbox_pred = self.rpn_bbox_pred(h)

        # enumerate all shifted anchors
        anchor = _enumerate_shifted_anchor(
            xp.array(self.anchor_base), self.feat_stride, ww, hh)
        roi = self.proposal_layer(
            rpn_bbox_pred, rpn_cls_prob, anchor, img_size,
            scale=scale, train=train)
        return rpn_bbox_pred, rpn_cls_score, roi, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, width, height):
    xp = cuda.get_array_module(anchor_base)
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
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
        shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
