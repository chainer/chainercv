import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from dummy_faster_rcnn import DummyFasterRCNN


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        self.n_anchor_base = 6
        self.feat_stride = 4
        n_fg_class = 4
        self.n_class = n_fg_class + 1
        self.n_roi = 24
        self.link = DummyFasterRCNN(
            n_anchor_base=self.n_anchor_base,
            feat_stride=self.feat_stride,
            n_fg_class=n_fg_class,
            n_roi=self.n_roi,
            min_size=600,
            max_size=1000,
        )

    def check_call(self):
        xp = self.link.xp

        x1 = chainer.Variable(_random_array(xp, (1, 3, 600, 800)))
        roi_cls_locs, roi_scores, rois, roi_indices = self.link(x1)

        self.assertIsInstance(roi_cls_locs, chainer.Variable)
        self.assertIsInstance(roi_cls_locs.data, xp.ndarray)
        self.assertEqual(roi_cls_locs.shape, (self.n_roi, self.n_class * 4))

        self.assertIsInstance(roi_scores, chainer.Variable)
        self.assertIsInstance(roi_scores.data, xp.ndarray)
        self.assertEqual(roi_scores.shape, (self.n_roi, self.n_class))

        self.assertIsInstance(rois, xp.ndarray)
        self.assertEqual(rois.shape, (self.n_roi, 4))

        self.assertIsInstance(roi_indices, xp.ndarray)
        self.assertEqual(roi_indices.shape, (self.n_roi,))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def check_predict(self):
        imgs = [
            _random_array(np, (3, 640, 480)),
            _random_array(np, (3, 320, 320))]

        bboxes, labels, scores = self.link.predict(imgs)

        self.assertEqual(len(bboxes), len(imgs))
        self.assertEqual(len(labels), len(imgs))
        self.assertEqual(len(scores), len(imgs))

        for bbox, label, score in zip(bboxes, labels, scores):
            self.assertIsInstance(bbox, np.ndarray)
            self.assertEqual(bbox.dtype, np.float32)
            self.assertEqual(bbox.ndim, 2)
            self.assertLessEqual(bbox.shape[0], self.n_roi)
            self.assertEqual(bbox.shape[1], 4)

            self.assertIsInstance(label, np.ndarray)
            self.assertEqual(label.dtype, np.int32)
            self.assertEqual(label.shape, (bbox.shape[0],))

            self.assertIsInstance(score, np.ndarray)
            self.assertEqual(score.dtype, np.float32)
            self.assertEqual(score.shape, (bbox.shape[0],))

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()


@testing.parameterize(
    {'in_shape': (3, 100, 100), 'expected_shape': (3, 200, 200)},
    {'in_shape': (3, 200, 50), 'expected_shape': (3, 400, 100)},
    {'in_shape': (3, 400, 100), 'expected_shape': (3, 400, 100)},
    {'in_shape': (3, 300, 600), 'expected_shape': (3, 200, 400)},
    {'in_shape': (3, 600, 900), 'expected_shape': (3, 200, 300)}
)
class TestFasterRCNNPrepare(unittest.TestCase):

    min_size = 200
    max_size = 400

    def setUp(self):
        self.link = DummyFasterRCNN(
            n_anchor_base=1,
            feat_stride=16,
            n_fg_class=21,
            n_roi=1,
            min_size=self.min_size,
            max_size=self.max_size
        )

    def check_prepare(self):
        x = _random_array(np, self.in_shape)
        out = self.link.prepare(x)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.expected_shape)

    def test_prepare_cpu(self):
        self.check_prepare()

    @attr.gpu
    def test_prepare_gpu(self):
        self.link.to_gpu()
        self.check_prepare()


testing.run_module(__name__, __file__)
