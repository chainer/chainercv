from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.fpn import BboxHead
from chainercv.links.model.fpn import FasterRCNN
from chainercv.links.model.fpn import MaskHead
from chainercv.links.model.fpn import RPN
from chainercv.utils import assert_is_bbox
from chainercv.utils import assert_is_detection_link
from chainercv.utils import assert_is_instance_segmentation_link


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

    def __init__(self, n_fg_class, return_values, min_size, max_size):
        extractor = DummyExtractor()
        super(DummyFasterRCNN, self).__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            bbox_head=BboxHead(n_fg_class + 1, extractor.scales),
            mask_head=MaskHead(n_fg_class + 1, extractor.scales),
            return_values=return_values,
            min_size=min_size, max_size=max_size,
        )


@testing.parameterize(*testing.product_dict(
    [
        {'return_values': 'detection'},
        {'return_values': 'instance_segmentation'},
        {'return_values': 'rpn'}
    ],
    [
        {'n_fg_class': 1},
        {'n_fg_class': 5},
        {'n_fg_class': 20},
    ],
    [
        # {
        #     'in_sizes': [(480, 640), (320, 320)],
        #     'min_size': 800, 'max_size': 1333,
        #     'expected_shape': (800, 1088),
        # },
        {
            'in_sizes': [(200, 50), (400, 100)],
            'min_size': 200, 'max_size': 320,
            'expected_shape': (320, 96),
        },
    ],
))
class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        if self.return_values == 'detection':
            return_values = ['bboxes', 'labels', 'scores']
        elif self.return_values == 'instance_segmentation':
            return_values = ['masks', 'labels', 'scores']
        elif self.return_values == 'rpn':
            return_values = ['rois']
        self.link = DummyFasterRCNN(n_fg_class=self.n_fg_class,
                                    return_values=return_values,
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
            hs, rois, roi_indices = self.link(x)

        self.assertEqual(len(hs), len(self.link.extractor.scales))
        for l in range(len(self.link.extractor.scales)):
            self.assertIsInstance(hs[l], chainer.Variable)
            self.assertIsInstance(hs[l].data, self.link.xp.ndarray)

        self.assertIsInstance(rois, self.link.xp.ndarray)
        self.assertEqual(rois.shape[1:], (4,))

        self.assertIsInstance(roi_indices, self.link.xp.ndarray)
        self.assertEqual(roi_indices.shape[1:], ())

        self.assertEqual(rois.shape[0], roi_indices.shape[0])

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

    def _check_predict(self):
        if self.return_values == 'detection':
            assert_is_detection_link(self.link, self.n_fg_class)
        elif self.return_values == 'instance_segmentation':
            assert_is_instance_segmentation_link(self.link, self.n_fg_class)
        elif self.return_values == 'rpn':
            imgs = [
                np.random.randint(
                    0, 256, size=(3, 480, 320)).astype(np.float32),
                np.random.randint(
                    0, 256, size=(3, 480, 320)).astype(np.float32)]
            result = self.link.predict(imgs)
            assert len(result) == 1
            assert len(result[0]) == 1
            for i in range(len(result[0])):
                roi = result[0][i]
                assert_is_bbox(roi)

    @attr.slow
    def test_predict_cpu(self):
        self._check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self._check_predict()

    def test_prepare(self):
        imgs = [_random_array(np, (3, s[0], s[1])) for s in self.in_sizes]
        out, scales = self.link.prepare(imgs)
        self.assertIsInstance(out, np.ndarray)
        full_expected_shape = (len(self.in_sizes), 3,
                               self.expected_shape[0],
                               self.expected_shape[1])
        self.assertEqual(out.shape, full_expected_shape)


testing.run_module(__name__, __file__)
