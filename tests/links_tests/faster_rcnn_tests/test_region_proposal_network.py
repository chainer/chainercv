import numpy as np
import unittest

from chainer import testing

from chainercv.links import RegionProposalNetwork


class TestRegionProposalNetwork(unittest.TestCase):

    def setUp(self):
        feat_stride = 4
        H = 8
        W = 12
        self.rpn = RegionProposalNetwork(feat_stride=feat_stride)
        self.x = np.random.uniform(size=(1, 16, H, W)).astype(np.float32)
        self.img_size = (W * feat_stride, H * feat_stride)

    def _check_call(self, x, img_size):
        rpn_bbox_pred, rpn_cls_prob, roi, anchor = self.rpn(x, img_size)

    def test_call_cpu(self):
        self._check_call(self.x, self.img_size)


testing.run_module(__name__, __file__)
