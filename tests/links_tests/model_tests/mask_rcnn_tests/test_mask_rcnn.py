from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.fpn import Head
from chainercv.links.model.fpn import RPN
from chainercv.links.model.mask_rcnn import MaskRCNN
from chainercv.links.model.mask_rcnn import MaskHead
from chainercv.utils import assert_is_instance_segmentation_link


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class DummyExtractor(chainer.Link):
    scales = (1 / 2, 1 / 4, 1 / 8)
    mean = _random_array(np, (3, 1, 1))
    n_channel = 16

    def __call__(self, x):
        n, _, h, w = x.shape
        return [chainer.Variable(_random_array(
                self.xp, (n, self.n_channel, int(h * scale), int(w * scale))))
                for scale in self.scales]


class DummyMaskRCNN(MaskRCNN):

    def __init__(self, n_fg_class):
        extractor = DummyExtractor()
        n_class = n_fg_class + 1
        super(DummyMaskRCNN, self).__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_class, extractor.scales),
            mask_head=MaskHead(n_class, extractor.scales)
        )


@testing.parameterize(
    {'n_fg_class': 1},
    {'n_fg_class': 5},
    {'n_fg_class': 20},
)
class TestMaskRCNN(unittest.TestCase):

    def setUp(self):
        self.link = DummyMaskRCNN(n_fg_class=self.n_fg_class)

    def test_use_preset(self):
        self.link.nms_thresh = 0
        self.link.score_thresh = 0

        self.link.use_preset('visualize')
        self.assertEqual(self.link.nms_thresh, 0.5)
        self.assertEqual(self.link.score_thresh, 0.7)

        self.link.nms_thresh = 0
        self.link.score_thresh = 0

        self.link.use_preset('evaluate')
        self.assertEqual(self.link.nms_thresh, 0.5)
        self.assertEqual(self.link.score_thresh, 0.05)

        with self.assertRaises(ValueError):
            self.link.use_preset('unknown')

    def _check_call(self):
        B = 2
        size = 32
        x = _random_array(self.link.xp, (B, 3, size, size))
        with chainer.using_config('train', False):
            hs, rois, roi_indices = self.link(x)

        self.assertEqual(len(hs), len(self.link.extractor.scales))
        self.assertEqual(len(rois), len(self.link.extractor.scales))
        self.assertEqual(len(roi_indices), len(self.link.extractor.scales))
        for l, scale in enumerate(self.link.extractor.scales):
            self.assertIsInstance(rois[l], self.link.xp.ndarray)
            self.assertEqual(rois[l].shape[1:], (4,))

            self.assertIsInstance(roi_indices[l], self.link.xp.ndarray)
            self.assertEqual(roi_indices[l].shape[1:], ())

            self.assertEqual(rois[l].shape[0], roi_indices[l].shape[0])

            self.assertIsInstance(hs[l], chainer.Variable)
            self.assertIsInstance(hs[l].array, self.link.xp.ndarray)
            feat_size = int(size * scale)
            self.assertEqual(
                hs[l].shape,
                (B, self.link.extractor.n_channel, feat_size, feat_size))

    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def test_call_train_mode(self):
        x = _random_array(self.link.xp, (2, 3, 32, 32))
        with self.assertRaises(AssertionError):
            with chainer.using_config('train', True):
                self.link(x)

    def test_predict_cpu(self):
        assert_is_instance_segmentation_link(self.link, self.n_fg_class)

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_instance_segmentation_link(self.link, self.n_fg_class)

    def test_prepare(self):
        imgs = [
            np.random.randint(0, 255, size=(3, 480, 640)).astype(np.float32),
            np.random.randint(0, 255, size=(3, 320, 320)).astype(np.float32),
        ]
        x, _, _ = self.link.prepare(imgs)
        self.assertEqual(x.shape, (2, 3, 800, 1088))


testing.run_module(__name__, __file__)
