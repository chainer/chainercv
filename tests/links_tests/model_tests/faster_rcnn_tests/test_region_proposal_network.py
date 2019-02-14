import numpy as np
import unittest

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.faster_rcnn import RegionProposalNetwork


@testing.parameterize(*(testing.product({
    'B': [1],
    'train': [True, False],
    'scales': [None, 1.0, 2.0, [1.0]],
}) + testing.product({
    'B': [2],
    'train': [True, False],
    'scales': [None, 1.0, 2.0, [1.0, 2.0]],
})))
class TestRegionProposalNetwork(unittest.TestCase):

    def setUp(self):
        feat_stride = 4
        C = 16
        H = 8
        W = 12
        self.proposal_creator_params = {
            'n_train_post_nms': 10,
            'n_test_post_nms': 5}
        self.ratios = [0.25, 4]
        self.anchor_scales = [2, 4]
        self.link = RegionProposalNetwork(
            in_channels=C, mid_channels=24,
            ratios=self.ratios, anchor_scales=self.anchor_scales,
            feat_stride=feat_stride,
            proposal_creator_params=self.proposal_creator_params
        )
        self.x = np.random.uniform(size=(self.B, C, H, W)).astype(np.float32)
        self.img_size = (H * feat_stride, W * feat_stride)

        chainer.config.train = self.train

    def _check_call(self, x, img_size, scales):
        _, _, H, W = x.shape
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.link(
            chainer.Variable(x), img_size, scales)
        self.assertIsInstance(rpn_locs, chainer.Variable)
        self.assertIsInstance(rpn_locs.array, type(x))
        self.assertIsInstance(rpn_scores, chainer.Variable)
        self.assertIsInstance(rpn_scores.array, type(x))

        A = len(self.ratios) * len(self.anchor_scales)
        self.assertEqual(rpn_locs.shape, (self.B, H * W * A, 4))
        self.assertEqual(rpn_scores.shape, (self.B, H * W * A, 2))

        if chainer.config.train:
            roi_size = self.proposal_creator_params[
                'n_train_post_nms']
        else:
            roi_size = self.proposal_creator_params[
                'n_test_post_nms']

        self.assertIsInstance(rois, type(x))
        self.assertIsInstance(roi_indices, type(x))
        self.assertLessEqual(rois.shape[0], self.B * roi_size)
        self.assertLessEqual(roi_indices.shape[0], self.B * roi_size)

        # Depending randomly generated bounding boxes, this is not true.
        if roi_indices.shape[0] == self.B * roi_size:
            for i in range(self.B):
                s = slice(i * roi_size, (i + 1) * roi_size)
                np.testing.assert_equal(
                    cuda.to_cpu(roi_indices[s]),
                    i * np.ones((roi_size,), dtype=np.int32))

        self.assertIsInstance(anchor, type(x))
        self.assertEqual(anchor.shape, (A * H * W, 4))

    def test_call_cpu(self):
        self._check_call(self.x, self.img_size, self.scales)

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call(
            chainer.backends.cuda.to_gpu(self.x), self.img_size, self.scales)


testing.run_module(__name__, __file__)
