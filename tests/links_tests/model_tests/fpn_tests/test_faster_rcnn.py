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

    def forward(self, x):
        n, _, h, w = x.shape
        return [chainer.Variable(_random_array(
                self.xp, (n, 16, int(h * scale), int(w * scale))))
                for scale in self.scales]


class DummyFasterRCNN(FasterRCNN):

    def __init__(self, n_fg_class, min_size, max_size):
        extractor = DummyExtractor()
        super(DummyFasterRCNN, self).__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_fg_class + 1, extractor.scales),
            min_size=min_size, max_size=max_size,
        )


@testing.parameterize(*testing.product_dict(
    [
        {'n_fg_class': 1},
        {'n_fg_class': 5},
        {'n_fg_class': 20},
    ],
    [
        {
            'in_sizes': [(480, 640), (320, 320)],
            'min_size': 800, 'max_size': 1333,
            'expected_shape': (800, 1088),
        },
        {
            'in_sizes': [(200, 50), (400, 100)],
            'min_size': 200, 'max_size': 320,
            'expected_shape': (320, 96),
        },
    ],
))
class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        self.link = DummyFasterRCNN(n_fg_class=self.n_fg_class,
                                    min_size=self.min_size,
                                    max_size=self.max_size)

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
        imgs = [_random_array(np, (3, s[0], s[1])) for s in self.in_sizes]
        out, scales = self.link.prepare(imgs)
        self.assertIsInstance(out, np.ndarray)
        full_expected_shape = (len(self.in_sizes), 3,
                               self.expected_shape[0],
                               self.expected_shape[1])
        self.assertEqual(out.shape, full_expected_shape)


testing.run_module(__name__, __file__)
