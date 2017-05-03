import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer import utils

from chainercv.links.faster_rcnn.anchor_target_layer import AnchorTargetLayer


def generate_bbox(n, img_size, min_length, max_length):
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


class TestAnchorTargetLayer(unittest.TestCase):

    img_size = (320, 240)
    rpn_batch_size = 256
    n_anchor_base = 9
    rpn_fg_fraction = 0.5
    rpn_bbox_inside_weight = (1., 0.9, 0.8, 0.7)

    def setUp(self):
        n_bbox = 8
        self.feat_size = (self.img_size[0] / 16, self.img_size[1] / 16)
        n_anchor = self.n_anchor_base * np.prod(self.feat_size)

        self.anchor = generate_bbox(n_anchor, self.img_size, 16, 200)
        self.bbox = generate_bbox(n_bbox, self.img_size, 16, 200)[None]
        self.anchor_target_layer = AnchorTargetLayer(
            self.rpn_batch_size, rpn_fg_fraction=self.rpn_fg_fraction,
            rpn_bbox_inside_weight=self.rpn_bbox_inside_weight
        )

    def check_anchor_target_layer(
            self, anchor_target_layer,
            bbox, anchor, feat_size, img_size):
        xp = cuda.get_array_module(bbox)

        bbox_target, label, bbox_inside_weight, bbox_outside_weight =\
            self.anchor_target_layer(bbox, anchor, feat_size, img_size)

        # Test types
        self.assertIsInstance(bbox_target, xp.ndarray)
        self.assertIsInstance(label, xp.ndarray)
        self.assertIsInstance(bbox_inside_weight, xp.ndarray)
        self.assertIsInstance(bbox_outside_weight, xp.ndarray)

        # Test shapes
        self.assertEqual(bbox_target.shape,
                         (1, 4 * self.n_anchor_base) + self.feat_size[::-1])
        self.assertEqual(label.shape,
                         (1, self.n_anchor_base) + self.feat_size[::-1])
        self.assertEqual(bbox_inside_weight.shape,
                         (1, 4 * self.n_anchor_base) + self.feat_size[::-1])
        self.assertEqual(bbox_outside_weight.shape,
                         (1, 4 * self.n_anchor_base) + self.feat_size[::-1])

        # Test ratio of foreground and background labels
        np.testing.assert_equal(
            cuda.to_cpu(utils.force_array(xp.sum(label >= 0))),
            self.rpn_batch_size)
        n_fg = cuda.to_cpu(utils.force_array(xp.sum(label == 1)))
        n_bg = cuda.to_cpu(utils.force_array(xp.sum(label == 0)))
        self.assertLessEqual(
            n_fg, self.rpn_batch_size * self.rpn_fg_fraction)
        self.assertLessEqual(n_bg, self.rpn_batch_size - n_fg)

        # Test bbox_inside_weight
        _bbox_inside_weight = bbox_inside_weight.reshape(
            (1, self.n_anchor_base, 4) + self.feat_size[::-1])
        _bbox_inside_weight = _bbox_inside_weight.transpose(0, 1, 3, 4, 2)
        bbox_inside_weight_masked = _bbox_inside_weight[label == 1]
        bbox_inside_weight_target = xp.tile(
            xp.array(self.rpn_bbox_inside_weight),
            bbox_inside_weight_masked.shape[0]).reshape(-1, 4)
        np.testing.assert_almost_equal(
            cuda.to_cpu(bbox_inside_weight_masked),
            cuda.to_cpu(bbox_inside_weight_target))

        # Test bbox_outside_weight
        _bbox_outside_weight = bbox_outside_weight.reshape(
            (1, self.n_anchor_base, 4) + self.feat_size[::-1])
        _bbox_outside_weight = _bbox_outside_weight.transpose(0, 1, 3, 4, 2)
        bbox_outside_weight_masked = _bbox_outside_weight[label >= 0]
        bbox_outside_weight_target = xp.ones_like(
            bbox_outside_weight_masked) / (n_fg + n_bg)
        np.testing.assert_almost_equal(
            cuda.to_cpu(bbox_outside_weight_masked),
            cuda.to_cpu(bbox_outside_weight_target))

    @condition.retry(3)
    def test_anchor_target_layer_cpu(self):
        self.check_anchor_target_layer(
            self.anchor_target_layer,
            self.bbox,
            self.anchor,
            self.feat_size,
            self.img_size)

    @attr.gpu
    @condition.retry(3)
    def test_anchor_target_layer_gpu(self):
        self.check_anchor_target_layer(
            self.anchor_target_layer,
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.anchor),
            self.feat_size,
            self.img_size)


testing.run_module(__name__, __file__)
