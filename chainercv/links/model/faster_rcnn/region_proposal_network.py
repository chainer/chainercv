import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

from chainercv.links.model.faster_rcnn.utils.generate_anchor_base import \
    generate_anchor_base
from chainercv.links.model.faster_rcnn.utils.proposal_creator import \
    ProposalCreator


class RegionProposalNetwork(chainer.Chain):

    """Region Proposal Networks introduced in Faster RCNN.

    This is Region Proposal Networks introduced in Faster RCNN [1].
    This takes features extracted from an image and predicts
    class agnostic bounding boxes around "objects".

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): Channel size of input.
        mid_channels (int): Channel size of the intermediate tensor.
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
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
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
                in_channels, mid_channels, 3, 1, 1, initialW=initializer),
            rpn_score=L.Convolution2D(
                mid_channels, 2 * n_anchor, 1, 1, 0, initialW=initializer),
            rpn_bbox=L.Convolution2D(
                mid_channels, 4 * n_anchor, 1, 1, 0, initialW=initializer)
        )

    def __call__(self, x, img_size, scale=1., train=False):
        """Forward Region Proposal Network.

        Currently, only arrays with batch size one are supported.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        An array of bounding boxes is an array of shape :math:`(R, 4)`, where
        :math:`R` is the number of  bounding boxes in an image. Each
        bouding box is organized by :obj:`(x_min, y_min, x_max, y_max)`
        in the second axis.

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
            (~chainer.Variable, ~chainer.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_bboxes**: Predicted regression targets for anchors. \
                Its shape is :math:`(1, 4 A, H, W)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(1, 2 A, H, W)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  The bounding box array is a concatenation of\
                bounding box arrays \
                from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes for the :math:`i` th image and size of batch \
                :math:`N`, :math:`R' = \\sum _{i=1} ^ N R_i`. \
                Each bouding box is organized by \
                :obj:`(x_min, y_min, x_max, y_max)` in the second axis. \
            * **batch_indices**: An array containing indices of images to \
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of anchors. This is an array of bounding\
                boxes. Its length is :math:`A`.

        """
        xp = cuda.get_array_module(x)
        n = x.data.shape[0]
        h = F.relu(self.rpn_conv_3x3(x))
        rpn_scores = self.rpn_score(h)
        c, hh, ww = rpn_scores.shape[1:]
        rpn_probs = F.softmax(rpn_scores.reshape(n, 2, -1))
        rpn_probs = rpn_probs.reshape(n, c, hh, ww)
        rpn_bboxes = self.rpn_bbox(h)

        # enumerate all shifted anchors
        anchor = _enumerate_shifted_anchor(
            xp.array(self.anchor_base), self.feat_stride, ww, hh)
        rois, batch_indices = self.proposal_layer(
            rpn_bboxes, rpn_probs, anchor, img_size,
            scale=scale, train=train)
        return rpn_bboxes, rpn_scores, rois, batch_indices, anchor


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
