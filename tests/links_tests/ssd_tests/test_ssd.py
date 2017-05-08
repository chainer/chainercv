import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.ssd import SSD


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class DummySSD(SSD):
    grids = (10, 4, 1)
    aspect_ratios = ((2,), (2, 3), (2,))
    steps = (0.1, 0.25, 1)
    sizes = (0.1, 0.25, 1, 1.2)

    def features(self, x):
        n_sample = x.shape[0]
        n_dims = (32, 16, 8)
        return [
            _random_array(self.xp, (n_sample, n_dim, grid, grid))
            for grid, n_dim in zip(self.grids, n_dims)]

    def prepare(self, img):
        _, H, W = img.shape
        return _random_array(self.xp, (3, 32, 32)), (W, H)


@testing.parameterize(*testing.product({
    'n_class': [1, 5, 20],
}))
class TestSSD(unittest.TestCase):

    def setUp(self):
        self.link = DummySSD(n_class=self.n_class)
        self.n_bbox = sum(
            grid * grid * ((len(ar) + 1) * 2)
            for grid, ar in zip(self.link.grids, self.link.aspect_ratios))

    def _check_default_bbox(self):
        self.assertIsInstance(self.link._default_bbox, self.link.xp.ndarray)
        self.assertEqual(self.link._default_bbox.shape, (self.n_bbox, 4))

    def test_default_bbox_cpu(self):
        self._check_default_bbox()

    @attr.gpu
    def test_default_bbox_gpu(self):
        self.link.to_gpu()
        self._check_default_bbox()

    def _check_decode(self):
        loc = _random_array(self.link.xp, (1, self.n_bbox, 4))
        conf = _random_array(self.link.xp, (1, self.n_bbox, self.n_class + 1))

        bboxes, scores = self.link._decode(loc, conf)

        self.assertIsInstance(bboxes, self.link.xp.ndarray)
        self.assertEqual(bboxes.shape, (1, self.n_bbox, 4))
        self.assertIsInstance(scores, self.link.xp.ndarray)
        self.assertEqual(scores.shape, (1, self.n_bbox, self.n_class + 1))

    def test_decode_cpu(self):
        self._check_decode()

    @attr.gpu
    def test_decode_gpu(self):
        self.link.to_gpu()
        self._check_decode()

    def _check_call(self):
        x = _random_array(self.link.xp, (1, 3, 32, 32))

        loc, conf = self.link(x)

        self.assertIsInstance(loc, chainer.Variable)
        self.assertIsInstance(loc.data, self.link.xp.ndarray)
        self.assertEqual(loc.shape, (1, self.n_bbox, 4))
        self.assertIsInstance(conf, chainer.Variable)
        self.assertIsInstance(conf.data, self.link.xp.ndarray)
        self.assertEqual(conf.shape, (1, self.n_bbox, self.n_class + 1))

    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def _check_suppress(self):
        raw_bbox = _random_array(self.link.xp, (self.n_bbox, 4))
        raw_score = _random_array(
            self.link.xp, (self.n_bbox, self.n_class + 1))

        bbox, label, score = self.link._suppress(raw_bbox, raw_score)

        self.assertIsInstance(bbox, self.link.xp.ndarray)
        self.assertEqual(bbox.dtype, np.float32)
        self.assertEqual(bbox.ndim, 2)
        self.assertLessEqual(bbox.shape[0], self.n_bbox * self.n_class)
        self.assertEqual(bbox.shape[1], 4)

        self.assertIsInstance(label, self.link.xp.ndarray)
        self.assertEqual(label.dtype, np.int32)
        self.assertEqual(label.ndim, 1)
        self.assertEqual(label.shape[0], bbox.shape[0])

        self.assertIsInstance(score, self.link.xp.ndarray)
        self.assertEqual(score.dtype, np.float32)
        self.assertEqual(score.ndim, 1)
        self.assertEqual(score.shape[0], bbox.shape[0])

    def test_suppress_cpu(self):
        self._check_suppress()

    @attr.gpu
    def test_suppress_gpu(self):
        self.link.to_gpu()
        self._check_suppress()

    def _check_predict(self):
        imgs = [
            _random_array(self.link.xp, (3, 640, 480)),
            _random_array(self.link.xp, (3, 320, 320))]

        bboxes, labels, scores = self.link.predict(imgs)

        self.assertEqual(len(bboxes), len(imgs))
        self.assertEqual(len(labels), len(imgs))
        self.assertEqual(len(scores), len(imgs))

        for bbox, label, score in zip(bboxes, labels, scores):
            self.assertIsInstance(bbox, self.link.xp.ndarray)
            self.assertEqual(bbox.dtype, np.float32)
            self.assertEqual(bbox.ndim, 2)
            self.assertLessEqual(bbox.shape[0], self.n_bbox * self.n_class)
            self.assertEqual(bbox.shape[1], 4)

            self.assertIsInstance(label, self.link.xp.ndarray)
            self.assertEqual(label.dtype, np.int32)
            self.assertEqual(label.ndim, 1)
            self.assertEqual(label.shape[0], bbox.shape[0])

            self.assertIsInstance(score, self.link.xp.ndarray)
            self.assertEqual(score.dtype, np.float32)
            self.assertEqual(score.ndim, 1)
            self.assertEqual(score.shape[0], bbox.shape[0])

    def test_predict_cpu(self):
        self._check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self._check_predict()


testing.run_module(__name__, __file__)
