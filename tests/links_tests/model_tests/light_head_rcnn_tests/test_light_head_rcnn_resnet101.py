import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import LightHeadRCNNResNet101


@testing.parameterize(
    {'train': False},
    {'train': True}
)
class TestLightHeadRCNNResNet101(unittest.TestCase):

    B = 1
    n_fg_class = 20
    n_class = n_fg_class + 1
    n_anchor = 9
    n_train_post_nms = 12
    n_test_post_nms = 8

    def setUp(self):
        proposal_creator_params = {
            'n_train_post_nms': self.n_train_post_nms,
            'n_test_post_nms': self.n_test_post_nms
        }
        self.link = LightHeadRCNNResNet101(
            self.n_fg_class, pretrained_model=None,
            proposal_creator_params=proposal_creator_params)

        chainer.config.train = self.train

    def check_call(self):
        xp = self.link.xp

        feat_size = (12, 16)
        x = chainer.Variable(
            xp.random.uniform(
                low=-1., high=1.,
                size=(self.B, 3, feat_size[0] * 16, feat_size[1] * 16)
            ).astype(np.float32))
        roi_cls_locs, roi_scores, rois, roi_indices = self.link(x)

        n_roi = roi_scores.shape[0]
        if self.train:
            self.assertGreaterEqual(self.B * self.n_train_post_nms, n_roi)
        else:
            self.assertGreaterEqual(self.B * self.n_test_post_nms * 2, n_roi)

        self.assertIsInstance(roi_cls_locs, chainer.Variable)
        self.assertIsInstance(roi_cls_locs.array, xp.ndarray)
        self.assertEqual(roi_cls_locs.shape, (n_roi, self.n_class * 4))

        self.assertIsInstance(roi_scores, chainer.Variable)
        self.assertIsInstance(roi_scores.array, xp.ndarray)
        self.assertEqual(roi_scores.shape, (n_roi, self.n_class))

        self.assertIsInstance(rois, xp.ndarray)
        self.assertEqual(rois.shape, (n_roi, 4))

        self.assertIsInstance(roi_indices, xp.ndarray)
        self.assertEqual(roi_indices.shape, (n_roi,))

    @attr.slow
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


@testing.parameterize(*testing.product({
    'n_fg_class': [None, 10, 20, 80],
    'anchor_scales': [(8, 16, 32), (4, 8, 16, 32), (2, 4, 8, 16, 32)],
    'pretrained_model': ['coco'],
}))
class TestLightHeadRCNNResNet101Pretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_fg_class': self.n_fg_class,
            'anchor_scales': self.anchor_scales,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model == 'coco':
            valid = self.n_fg_class in [None, 80]
            valid = valid and self.anchor_scales == (2, 4, 8, 16, 32)

        if valid:
            LightHeadRCNNResNet101(**kwargs)
        else:
            with self.assertRaises(ValueError):
                LightHeadRCNNResNet101(**kwargs)


testing.run_module(__name__, __file__)
