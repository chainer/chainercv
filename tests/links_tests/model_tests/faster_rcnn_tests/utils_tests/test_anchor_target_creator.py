import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer import utils

from chainercv.links.model.faster_rcnn import AnchorTargetCreator


def _generate_bbox(n, img_size, min_length, max_length):
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


class TestAnchorTargetCreator(unittest.TestCase):

    img_size = (320, 240)
    n_sample = 256
    n_anchor_base = 9
    fg_fraction = 0.5
    loc_in_weight_base = (1., 0.9, 0.8, 0.7)

    def setUp(self):
        n_bbox = 8
        feat_size = (self.img_size[0] // 16, self.img_size[1] // 16)
        self.n_anchor = self.n_anchor_base * np.prod(feat_size)

        self.anchor = _generate_bbox(self.n_anchor, self.img_size, 16, 200)
        self.raw_bbox = _generate_bbox(n_bbox, self.img_size, 16, 200)
        self.anchor_target_layer = AnchorTargetCreator(
            self.n_sample, fg_fraction=self.fg_fraction,
            loc_in_weight_base=self.loc_in_weight_base
        )

    def check_anchor_target_creator(
            self, anchor_target_layer,
            raw_bbox, anchor, img_size):
        xp = cuda.get_array_module(raw_bbox)

        bbox_target, label, loc_in_weight, bbox_out_weight =\
            self.anchor_target_layer(raw_bbox, anchor, img_size)

        # Test types
        self.assertIsInstance(bbox_target, xp.ndarray)
        self.assertIsInstance(label, xp.ndarray)
        self.assertIsInstance(loc_in_weight, xp.ndarray)
        self.assertIsInstance(bbox_out_weight, xp.ndarray)

        # Test shapes
        self.assertEqual(bbox_target.shape, (self.n_anchor, 4))
        self.assertEqual(label.shape, (self.n_anchor,))
        self.assertEqual(loc_in_weight.shape, (self.n_anchor, 4))
        self.assertEqual(bbox_out_weight.shape, (self.n_anchor, 4))

        # Test dtype
        self.assertEqual(bbox_target.dtype, np.float32)
        self.assertEqual(label.dtype, np.int32)
        self.assertEqual(loc_in_weight.dtype, np.float32)
        self.assertEqual(bbox_out_weight.dtype, np.float32)

        # Test ratio of foreground and background labels
        np.testing.assert_equal(
            cuda.to_cpu(utils.force_array(xp.sum(label >= 0))),
            self.n_sample)
        n_fg = cuda.to_cpu(utils.force_array(xp.sum(label == 1)))
        n_bg = cuda.to_cpu(utils.force_array(xp.sum(label == 0)))
        self.assertLessEqual(
            n_fg, self.n_sample * self.fg_fraction)
        self.assertLessEqual(n_bg, self.n_sample - n_fg)

        # Test loc_in_weight
        loc_in_weight_masked = loc_in_weight[label == 1]
        loc_in_weight_target = xp.tile(
            xp.array(self.loc_in_weight_base),
            loc_in_weight_masked.shape[0]).reshape(-1, 4)
        np.testing.assert_almost_equal(
            cuda.to_cpu(loc_in_weight_masked),
            cuda.to_cpu(loc_in_weight_target))

        # # Test bbox_out_weight
        bbox_out_weight_masked = bbox_out_weight[label >= 0]
        bbox_out_weight_target = xp.ones_like(
            bbox_out_weight_masked) / (n_fg + n_bg)
        np.testing.assert_almost_equal(
            cuda.to_cpu(bbox_out_weight_masked),
            cuda.to_cpu(bbox_out_weight_target))

    def test_anchor_target_creator_cpu(self):
        self.check_anchor_target_creator(
            self.anchor_target_layer,
            self.raw_bbox,
            self.anchor,
            self.img_size)

    @attr.gpu
    def test_anchor_target_creator_gpu(self):
        self.check_anchor_target_creator(
            self.anchor_target_layer,
            cuda.to_gpu(self.raw_bbox),
            cuda.to_gpu(self.anchor),
            self.img_size)


testing.run_module(__name__, __file__)
