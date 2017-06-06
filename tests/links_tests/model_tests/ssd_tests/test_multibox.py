import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import Multibox


@testing.parameterize(*testing.product({
    'n_class': [1, 5],
    'aspect_ratios': [((2,),), ((2,), (3, 4), (5,))],
    'batchsize': [1, 2],
}))
class TestMultibox(unittest.TestCase):

    def setUp(self):
        self.link = Multibox(self.n_class, self.aspect_ratios)

        xs = list()
        n_bbox = 0
        for ar in self.aspect_ratios:
            C, H, W = np.random.randint(1, 10, size=3)
            xs.append(
                np.random.uniform(size=(self.batchsize, C, H, W))
                .astype(np.float32))
            n_bbox += H * W * (len(ar) + 1) * 2

        self.xs = xs
        self.n_bbox = n_bbox

    def _check_forward(self, xs):
        loc, conf = self.link(xs)

        self.assertIsInstance(loc, chainer.Variable)
        self.assertIsInstance(loc.data, type(xs[0]))
        self.assertEqual(loc.shape, (self.batchsize, self.n_bbox, 4))
        self.assertEqual(loc.dtype, xs[0].dtype)

        self.assertIsInstance(conf, chainer.Variable)
        self.assertIsInstance(conf.data, type(xs[0]))
        self.assertEqual(
            conf.shape, (self.batchsize, self.n_bbox, self.n_class))
        self.assertEqual(conf.dtype, xs[0].dtype)

    def test_forward_cpu(self):
        self._check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self._check_forward(list(map(chainer.cuda.to_gpu, self.xs)))


testing.run_module(__name__, __file__)
