import numpy as np
import unittest

import chainer

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
        self.roi_prob = np.random.uniform(size=(n_roi, n_class)).astype(np.float32)
        self.bbox = generate_random_bbox(n_roi, self.size, 0, 18)

    def check_mask_voting(
            self, roi_mask_prob, bbox, roi_prob,
            size, bg_label, roi_size):
        xp = chainer.cuda.get_array_module(roi_mask_prob)
        roi_mask_prob, bbox, label, score = mask_voting(
            roi_mask_prob, bbox, roi_prob, size,
            0.5, 0.3, 0.5, 0.4, bg_label=bg_label)

        n_roi = roi_mask_prob.shape[0]
        self.assertIsInstance(roi_mask_prob, xp.ndarray)
        self.assertEqual(roi_mask_prob.shape[1:], (roi_size, roi_size))
        self.assertTrue(
            xp.all(xp.logical_and(roi_mask_prob >= 0.0, roi_mask_prob <= 1.0)))

        self.assertIsInstance(label, xp.ndarray)
        self.assertEqual(label.shape, (n_roi, ))

        self.assertIsInstance(score, xp.ndarray)
        self.assertEqual(score.shape, (n_roi, ))

        assert_is_bbox(bbox, size)

    def test_mask_voting_cpu(self):
        self.check_mask_voting(
            self.roi_mask_prob, self.bbox, self.roi_prob,
            self.size, self.bg_label, self.roi_size)
