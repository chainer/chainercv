import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import SSD300
from chainercv.links import SSD512


@testing.parameterize(*testing.product({
    'insize': [300, 512],
    'n_classes': [1, 5, 20],
}))
class TestSSD(unittest.TestCase):

    def setUp(self):
        if self.insize == 300:
            self.link = SSD300(n_classes=self.n_classes, pretrained_model=None)
            self.n_bbox = 8732
        elif self.insize == 512:
            self.link = SSD512(n_classes=self.n_classes, pretrained_model=None)
            self.n_bbox = 24564

    def _random_array(self, shape):
        return self.link.xp.array(
            np.random.uniform(-1, 1, size=shape), dtype=np.float32)

    def _random_image(self, width, height):
        return np.random.randint(0, 255, size=(3, height, width))

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
        loc = self._random_array((1, self.n_bbox, 4))
        conf = self._random_array((1, self.n_bbox, self.n_classes + 1))

        bboxes, scores = self.link._decode(loc, conf)

        self.assertIsInstance(bboxes, self.link.xp.ndarray)
        self.assertEqual(bboxes.shape, (1, self.n_bbox, 4))
        self.assertIsInstance(scores, self.link.xp.ndarray)
        self.assertEqual(scores.shape, (1, self.n_bbox, self.n_classes + 1))

    def test_decode_cpu(self):
        self._check_decode()

    @attr.gpu
    def test_decode_gpu(self):
        self.link.to_gpu()
        self._check_decode()

    def _check_call(self):
        x = self._random_array((1, 3, self.insize, self.insize))

        loc, conf = self.link(x)

        self.assertIsInstance(loc, chainer.Variable)
        self.assertIsInstance(loc.data, self.link.xp.ndarray)
        self.assertEqual(loc.shape, (1, self.n_bbox, 4))
        self.assertIsInstance(conf, chainer.Variable)
        self.assertIsInstance(conf.data, self.link.xp.ndarray)
        self.assertEqual(conf.shape, (1, self.n_bbox, self.n_classes + 1))

    @attr.slow
    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def test_prepare(self):
        img = self._random_image(640, 480)
        img, size = self.link._prepare(img)
        self.assertEqual(img.shape, (3, self.insize, self.insize))
        self.assertEqual(size, (640, 480))

    def _check_predict(self):
        imgs = [
            self._random_image(640, 480),
            self._random_image(320, 320)]

        with np.errstate(divide='ignore'):
            bboxes, labels, scores = self.link.predict(imgs)

        self.assertEqual(len(bboxes), len(imgs))
        self.assertEqual(len(labels), len(imgs))
        self.assertEqual(len(scores), len(imgs))

        for bbox, label, score in zip(bboxes, labels, scores):
            self.assertIsInstance(bbox, self.link.xp.ndarray)
            self.assertEqual(bbox.dtype, np.float32)
            self.assertEqual(bbox.ndim, 2)
            self.assertLessEqual(bbox.shape[0], self.n_bbox * self.n_classes)
            self.assertEqual(bbox.shape[1], 4)

            self.assertIsInstance(label, self.link.xp.ndarray)
            self.assertEqual(label.dtype, np.int32)
            self.assertEqual(label.ndim, 1)
            self.assertEqual(label.shape[0], bbox.shape[0])

            self.assertIsInstance(score, self.link.xp.ndarray)
            self.assertEqual(score.dtype, np.float32)
            self.assertEqual(score.ndim, 1)
            self.assertEqual(score.shape[0], bbox.shape[0])

    @attr.slow
    def test_predict_cpu(self):
        self._check_predict()

    @attr.gpu
    @attr.slow
    def test_predict_gpu(self):
        self.link.to_gpu()
        self._check_predict()


testing.run_module(__name__, __file__)
