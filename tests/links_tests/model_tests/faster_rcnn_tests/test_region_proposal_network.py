import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import RegionProposalNetwork


@testing.parameterize(
    {'train': True},
    {'train': False},
)
class TestRegionProposalNetwork(unittest.TestCase):

    def setUp(self):
        feat_stride = 4
        C = 16
        H = 8
        W = 12
        self.proposal_creator_params = {
            'train_rpn_post_nms_top_n': 10,
            'test_rpn_post_nms_top_n': 5}
        self.ratios = [0.25, 4]
        self.anchor_scales = [2, 4]
        self.link = RegionProposalNetwork(
            n_in=C, n_mid=24,
            ratios=self.ratios, anchor_scales=self.anchor_scales,
            feat_stride=feat_stride,
            proposal_creator_params=self.proposal_creator_params
        )
        self.x = np.random.uniform(size=(1, C, H, W)).astype(np.float32)
        self.img_size = (W * feat_stride, H * feat_stride)

    def _check_call(self, x, img_size, train):
        _, _, H, W = x.shape
        rpn_bbox_pred, rpn_cls_prob, proposals, anchor = self.link(
            chainer.Variable(x), img_size, train=train)
        self.assertIsInstance(rpn_bbox_pred, chainer.Variable)
        self.assertIsInstance(rpn_bbox_pred.data, type(x))
        self.assertIsInstance(rpn_cls_prob, chainer.Variable)
        self.assertIsInstance(rpn_cls_prob.data, type(x))

        A = len(self.ratios) * len(self.anchor_scales)
        self.assertEqual(rpn_bbox_pred.shape, (1, A * 4, H, W))
        self.assertEqual(rpn_cls_prob.shape, (1, A * 2, H, W))

        self.assertIsInstance(proposals, list)
        self.assertIsInstance(proposals[0], type(x))
        if train:
            roi_size = self.proposal_creator_params[
                'train_rpn_post_nms_top_n']
        else:
            roi_size = self.proposal_creator_params[
                'test_rpn_post_nms_top_n']
        self.assertEqual(proposals[0].shape, (roi_size, 4))

        self.assertIsInstance(anchor, type(x))
        self.assertEqual(anchor.shape, (A * H * W, 4))

    def test_call_cpu(self):
        self._check_call(self.x, self.img_size, self.train)

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call(
            chainer.cuda.to_gpu(self.x), self.img_size, self.train)


testing.run_module(__name__, __file__)
