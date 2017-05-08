import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import SSD300
from chainercv.links import SSD512


@testing.parameterize(*testing.product({
    'insize': [300, 512],
    'n_class': [1, 5, 20],
}))
class TestSSD(unittest.TestCase):

    def setUp(self):
        if self.insize == 300:
            self.link = SSD300(n_class=self.n_class)
            self.n_bbox = 8732
        elif self.insize == 512:
            self.link = SSD512(n_class=self.n_class)
            self.n_bbox = 24564

    def _check_call(self):
        x = self.link.xp.array(
            np.random.uniform(-1, 1, size=(1, 3, self.insize, self.insize)),
            dtype=np.float32)

        loc, conf = self.link(x)

        self.assertIsInstance(loc, chainer.Variable)
        self.assertIsInstance(loc.data, self.link.xp.ndarray)
        self.assertEqual(loc.shape, (1, self.n_bbox, 4))
        self.assertIsInstance(conf, chainer.Variable)
        self.assertIsInstance(conf.data, self.link.xp.ndarray)
        self.assertEqual(conf.shape, (1, self.n_bbox, self.n_class + 1))

    @attr.slow
    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def test_prepare(self):
        img = np.random.randint(0, 255, size=(3, 480, 640))
        img = self.link.prepare(img)
        self.assertEqual(img.shape, (3, self.insize, self.insize))


testing.run_module(__name__, __file__)
