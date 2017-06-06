import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer import utils

from chainercv.links.model.faster_rcnn import AnchorTargetCreator
from chainercv.utils import generate_random_bbox


class TestAnchorTargetCreator(unittest.TestCase):

    img_size = (320, 240)
    n_sample = 256
    n_anchor_base = 9
    pos_ratio = 0.5

    def setUp(self):
        n_bbox = 8
        feat_size = (self.img_size[0] // 16, self.img_size[1] // 16)
        self.n_anchor = self.n_anchor_base * np.prod(feat_size)

        self.anchor = generate_random_bbox(
            self.n_anchor, self.img_size, 16, 200)
        self.bbox = generate_random_bbox(n_bbox, self.img_size, 16, 200)
        self.anchor_target_layer = AnchorTargetCreator(
            self.n_sample, pos_ratio=self.pos_ratio,
        )

    def check_anchor_target_creator(
            self, anchor_target_layer,
            bbox, anchor, img_size):
        xp = cuda.get_array_module(bbox)

        loc, label = self.anchor_target_layer(
            bbox, anchor, img_size)

        # Test types
        self.assertIsInstance(loc, xp.ndarray)
        self.assertIsInstance(label, xp.ndarray)

        # Test shapes
        self.assertEqual(loc.shape, (self.n_anchor, 4))
        self.assertEqual(label.shape, (self.n_anchor,))

        # Test dtype
        self.assertEqual(loc.dtype, np.float32)
        self.assertEqual(label.dtype, np.int32)

        # Test ratio of foreground and background labels
        np.testing.assert_equal(
            cuda.to_cpu(utils.force_array(xp.sum(label >= 0))),
            self.n_sample)
        n_pos = cuda.to_cpu(utils.force_array(xp.sum(label == 1)))
        n_neg = cuda.to_cpu(utils.force_array(xp.sum(label == 0)))
        self.assertLessEqual(
            n_pos, self.n_sample * self.pos_ratio)
        self.assertLessEqual(n_neg, self.n_sample - n_pos)

    def test_anchor_target_creator_cpu(self):
        self.check_anchor_target_creator(
            self.anchor_target_layer,
            self.bbox,
            self.anchor,
            self.img_size)

    @attr.gpu
    def test_anchor_target_creator_gpu(self):
        self.check_anchor_target_creator(
            self.anchor_target_layer,
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.anchor),
            self.img_size)


testing.run_module(__name__, __file__)
