from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.fpn import FasterRCNN
from chainercv.links.model.fpn import Head
from chainercv.links.model.fpn import RPN
from chainercv.utils import assert_is_detection_link


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class DummyExtractor(chainer.Link):
    scales = (1 / 2, 1 / 4, 1 / 8)
    mean = _random_array(np, (3, 1, 1))

    def __call__(self, x):
        n, _, h, w = x.shape
        return [chainer.Variable(_random_array(
                self.xp, (n, 16, int(h * scale), int(w * scale))))
                for scale in self.scales]


class DummyFasterRCNN(FasterRCNN):

    def __init__(self, n_fg_class):
        extractor = DummyExtractor()
        super(DummyFasterRCNN, self).__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_fg_class + 1, extractor.scales),
        )


@testing.parameterize(
    {'n_fg_class': 1},
    {'n_fg_class': 5},
    {'n_fg_class': 20},
)
class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        self.link = DummyFasterRCNN(n_fg_class=self.n_fg_class)

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
        x = _random_array(self.link.xp, (2, 3, 32, 32))
        with chainer.using_config('train', False):
            rois, roi_indices, head_locs, head_confs = self.link(x)

        self.assertEqual(len(rois), len(self.link.extractor.scales))
        self.assertEqual(len(roi_indices), len(self.link.extractor.scales))
        for l in range(len(self.link.extractor.scales)):
            self.assertIsInstance(rois[l], self.link.xp.ndarray)
            self.assertEqual(rois[l].shape[1:], (4,))

            self.assertIsInstance(roi_indices[l], self.link.xp.ndarray)
            self.assertEqual(roi_indices[l].shape[1:], ())

            self.assertEqual(rois[l].shape[0], roi_indices[l].shape[0])

        n_roi = sum(
            len(rois[l]) for l in range(len(self.link.extractor.scales)))

        self.assertIsInstance(head_locs, chainer.Variable)
        self.assertIsInstance(head_locs.array, self.link.xp.ndarray)
        self.assertEqual(head_locs.shape, (n_roi, self.n_fg_class + 1, 4))

        self.assertIsInstance(head_confs, chainer.Variable)
        self.assertIsInstance(head_confs.array, self.link.xp.ndarray)
        self.assertEqual(head_confs.shape, (n_roi, self.n_fg_class + 1))

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
        assert_is_detection_link(self.link, self.n_fg_class)

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_detection_link(self.link, self.n_fg_class)

    def test_prepare(self):
        imgs = [
            np.random.randint(0, 255, size=(3, 480, 640)).astype(np.float32),
            np.random.randint(0, 255, size=(3, 480, 640)).astype(np.uint8),
            np.random.randint(0, 255, size=(3, 320, 320)).astype(np.float32),
            np.random.randint(0, 255, size=(3, 320, 320)).astype(np.uint8),
        ]
        x, scales = self.link.prepare(imgs)
        self.assertEqual(x.shape, (len(imgs), 3, 800, 1088))


testing.run_module(__name__, __file__)
