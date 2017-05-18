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
    This takes features extracted from images and predicts
    class agnostic bounding boxes around "objects".

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): Anchors with ratios contained in this list
            will be generated. Ratio is the height divided by the width.
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
            conv1=L.Convolution2D(
                in_channels, mid_channels, 3, 1, 1, initialW=initializer),
            score=L.Convolution2D(
                mid_channels, 2 * n_anchor, 1, 1, 0, initialW=initializer),
            bbox=L.Convolution2D(
                mid_channels, 4 * n_anchor, 1, 1, 0, initialW=initializer)
        )

    def __call__(self, x, img_size, scale=1., train=False):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

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

            * **rpn_bboxes**: Predicted bounding box offsets for anchors. \
                Its shape is :math:`(N, 4 A, H, W)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, 2 A, H, W)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`. \
            * **batch_indices**: An array containing indices of images to \
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        h = F.relu(self.conv1(x))
        rpn_scores = self.score(h)
        n, c, hh, ww = rpn_scores.shape
        rpn_probs = F.softmax(rpn_scores.reshape(n, 2, -1))
        rpn_probs = rpn_probs.reshape(n, c, hh, ww)
        rpn_bboxes = self.bbox(h)

        anchor = _enumerate_shifted_anchor(
            self.xp.array(self.anchor_base), self.feat_stride, ww, hh)
        rpn_bboxes_reshaped = rpn_bboxes.transpose(
            (0, 2, 3, 1)).reshape(n, -1, 4).data
        n_anchor = c // 2
        rpn_fg_probs = rpn_probs[:, n_anchor:].data
        rpn_fg_probs = rpn_fg_probs.transpose(0, 2, 3, 1).reshape(n, -1)

        rois = []
        batch_indices = []
        for i in range(n):
            roi = self.proposal_layer(
                rpn_bboxes_reshaped[i], rpn_fg_probs[i], anchor, img_size,
                scale=scale, train=train)
            batch_index = i * self.xp.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            batch_indices.append(batch_index)

        rois = self.xp.concatenate(rois, axis=0)
        batch_indices = self.xp.concatenate(batch_indices, axis=0)
        return rpn_bboxes, rpn_scores, rois, batch_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, width, height):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    xp = cuda.get_array_module(anchor_base)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
        shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
