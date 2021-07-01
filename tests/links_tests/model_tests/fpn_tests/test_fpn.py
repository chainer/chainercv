from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.fpn import FPN


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class DummyExtractor(chainer.Link):
    mean = _random_array(np, (3, 1, 1))

    def forward(self, x):
        n, _, h, w = x.shape
        return [
            chainer.Variable(_random_array(self.xp, (n, 16, h // 2, w // 2))),
            chainer.Variable(_random_array(self.xp, (n, 32, h // 4, w // 4))),
            chainer.Variable(_random_array(self.xp, (n, 64, h // 8, w // 8))),
        ]


class TestFPN(unittest.TestCase):

    def setUp(self):
        self.link = FPN(
            base=DummyExtractor(),
            n_base_output=3,
            scales=(1 / 2, 1 / 4, 1 / 8))

    def test_mean(self):
        np.testing.assert_equal(self.link.mean, self.link.base.mean)

    def _check_call(self):
        x = _random_array(self.link.xp, (2, 3, 32, 32))
        hs = self.link(x)

        self.assertEqual(len(hs), 3)
        for l in range(len(hs)):
            self.assertIsInstance(hs[l], chainer.Variable)
            self.assertIsInstance(hs[l].array, self.link.xp.ndarray)
            self.assertEqual(hs[l].shape, (2, 256, 16 >> l, 16 >> l))

    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()


testing.run_module(__name__, __file__)
