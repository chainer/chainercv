import numpy as np
import unittest

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

    def _random_array(self, *shape):
        xp = self.link.xp
        return xp.array(np.random.uniform(-1, 1, shape), dtype=np.float32)

    @attr.slow
    def test_call(self):

        x = self._random_array(1, 3, self.insize, self.insize)
        loc, conf = self.link(x)
        self.assertEqual(loc.shape, (1, self.n_bbox, 4))
        self.assertEqual(conf.shape, (1, self.n_bbox, self.n_classes + 1))

    def test_default_bbox(self):
        self.assertEqual(self.link.default_bbox.shape, (self.n_bbox, 4))

    def test_decode(self):
        loc = self._random_array(self.n_bbox, 4)
        conf = self._random_array(self.n_bbox, self.n_classes + 1)
        bbox, score = self.link._decode(loc, conf)
        self.assertEqual(bbox.shape, (self.n_bbox, 4))
        self.assertEqual(score.shape, (self.n_bbox, self.n_classes + 1))

    def test_prepare(self):
        img = self._random_array(3, 480, 640)
        img, size = self.link._prepare(img)
        self.assertEqual(img.shape, (3, self.insize, self.insize))
        self.assertEqual(size, (640, 480))

    @attr.slow
    def test_predict(self):
        img = self._random_array(3, 480, 640)
        with np.errstate(divide='ignore'):
            bbox, label, score = self.link.predict(img)
        self.assertLessEqual(bbox.shape[0], self.n_bbox * self.n_classes)
        self.assertEqual(bbox.shape[0], label.shape[0])
        self.assertEqual(bbox.shape[0], score.shape[0])
        self.assertEqual(bbox.shape[1:], (4,))
        self.assertEqual(label.shape[1:], (1,))
        self.assertEqual(score.shape[1:], (1,))


testing.run_module(__name__, __file__)
