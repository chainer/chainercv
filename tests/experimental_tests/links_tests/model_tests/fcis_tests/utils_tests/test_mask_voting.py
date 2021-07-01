import numpy as np
import unittest

import chainer
from chainer import testing

from chainercv.experimental.links.model.fcis.utils.mask_voting \
    import mask_voting
from chainercv.utils import assert_is_bbox
from chainercv.utils import generate_random_bbox


class TestMaskVoting(unittest.TestCase):

    def setUp(self):
        n_roi = 5
        n_class = 6
        self.roi_size = 7
        self.size = (18, 24)
        self.bg_label = 0
        self.roi_mask_prob = np.random.uniform(
            size=(n_roi, self.roi_size, self.roi_size)).astype(np.float32)
        self.roi_prob = np.random.uniform(
            size=(n_roi, n_class)).astype(np.float32)
        self.bbox = generate_random_bbox(n_roi, self.size, 0, 18)

    def check_mask_voting(
            self, seg_prob, bbox, cls_prob,
            size, bg_label, roi_size):
        xp = chainer.cuda.get_array_module(seg_prob)
        seg_prob, bbox, label, cls_prob = mask_voting(
            seg_prob, bbox, cls_prob, size,
            0.5, 0.3, 0.5, 0.4, bg_label=bg_label)

        n_roi = seg_prob.shape[0]
        self.assertIsInstance(seg_prob, xp.ndarray)
        self.assertEqual(seg_prob.shape[1:], (roi_size, roi_size))
        self.assertTrue(
            xp.all(xp.logical_and(seg_prob >= 0.0, seg_prob <= 1.0)))

        self.assertIsInstance(label, xp.ndarray)
        self.assertEqual(label.shape, (n_roi, ))

        self.assertIsInstance(cls_prob, xp.ndarray)
        self.assertEqual(cls_prob.shape, (n_roi, ))

        assert_is_bbox(bbox, size)

    def test_mask_voting_cpu(self):
        self.check_mask_voting(
            self.roi_mask_prob, self.bbox, self.roi_prob,
            self.size, self.bg_label, self.roi_size)


testing.run_module(__name__, __file__)
