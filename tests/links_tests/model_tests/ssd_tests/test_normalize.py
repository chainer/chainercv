import numpy as np
import unittest

import chainer
from chainer import initializers
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import Normalize


@testing.parameterize(*testing.product({
    'shape': [(5, 5), (25, 25), (5, 25)],
    'n_channel': [1, 10],
    'eps': [1e-5, 1],
}))
class TestNormalize(unittest.TestCase):

    def setUp(self):
        self.link = Normalize(
            self.n_channel, initializers.Normal(), eps=self.eps)
        self.x = np.random.uniform(size=(1, self.n_channel) + self.shape) \
                          .astype(np.float32)

    def _check_forward(self, x):
        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, type(x))
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)

        x = chainer.cuda.to_cpu(x)
        y = chainer.cuda.to_cpu(y.data)
        scale = chainer.cuda.to_cpu(self.link.scale.data)

        norm = np.linalg.norm(x, axis=1, keepdims=True) + self.eps
        expect = x / norm * scale[:, np.newaxis, np.newaxis]
        np.testing.assert_almost_equal(y, expect)

    def test_forward_cpu(self):
        self._check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self._check_forward(chainer.cuda.to_gpu(self.x))

    def test_forward_zero_cpu(self):
        self._check_forward(np.zeros_like(self.x))

    @attr.gpu
    def test_forward_zero__gpu(self):
        self.link.to_gpu()
        self._check_forward(chainer.cuda.to_gpu(np.zeros_like(self.x)))


testing.run_module(__name__, __file__)
