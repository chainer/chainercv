import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links import FCISResNet101


@testing.parameterize(
    {'train': False, 'iter2': True},
    {'train': True, 'iter2': False}
)
class TestFCISResNet101(unittest.TestCase):

    B = 1
    n_fg_class = 20
    n_class = n_fg_class + 1
    n_anchor = 9
    n_train_post_nms = 12
    n_test_post_nms = 8

    def setUp(self):
        proposal_creator_params = {
            'n_train_post_nms': self.n_train_post_nms,
            'n_test_post_nms': self.n_test_post_nms,
        }
        self.link = FCISResNet101(
            self.n_fg_class, pretrained_model=None,
            iter2=self.iter2,
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
        roi_seg_scores, roi_ag_locs, roi_scores, rois, roi_indices = \
            self.link(x)

        n_roi = roi_seg_scores.shape[0]
        if self.train:
            self.assertGreaterEqual(self.B * self.n_train_post_nms, n_roi)
        else:
            self.assertGreaterEqual(self.B * self.n_test_post_nms * 2, n_roi)

        self.assertIsInstance(roi_seg_scores, chainer.Variable)
        self.assertIsInstance(roi_seg_scores.array, xp.ndarray)
        self.assertEqual(
            roi_seg_scores.shape, (n_roi, 2, 21, 21))

        self.assertIsInstance(roi_ag_locs, chainer.Variable)
        self.assertIsInstance(roi_ag_locs.array, xp.ndarray)
        self.assertEqual(roi_ag_locs.shape, (n_roi, 2, 4))

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


class TestFCISResNet101Pretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        FCISResNet101(pretrained_model='sbd')

    @attr.slow
    def test_pretrained_n_fg_class(self):
        FCISResNet101(n_fg_class=20, pretrained_model='sbd')

    @attr.slow
    def test_pretrained_wrong_n_fg_class(self):
        with self.assertRaises(ValueError):
            FCISResNet101(n_fg_class=10, pretrained_model='sbd')
