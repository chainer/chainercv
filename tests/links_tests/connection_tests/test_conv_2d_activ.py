import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer.functions import relu
from chainer import testing
from chainer.testing import attr

from chainercv.links import Conv2DActiv


def _add_one(x):
    return x + 1


@testing.parameterize(*testing.product({
    'args_style': ['explicit', 'None', 'omit'],
    'activ': ['relu', 'add_one']
}))
class TestConv2DActiv(unittest.TestCase):

    in_channels = 1
    out_channels = 1
    ksize = 3
    stride = 1
    pad = 1

    def setUp(self):
        if self.activ == 'relu':
            activ = relu
        elif self.activ == 'add_one':
            activ = _add_one
        self.x = np.random.uniform(
            -1, 1, (5, self.in_channels, 5, 5)).astype(np.float32)
        self.gy = np.random.uniform(
            -1, 1, (5, self.out_channels, 5, 5)).astype(np.float32)

        # Convolution is the identity function.
        initialW = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                            dtype=np.float32).reshape((1, 1, 3, 3))
        initial_bias = 0
        if self.args_style == 'explicit':
            self.l = Conv2DActiv(
                self.in_channels, self.out_channels, self.ksize,
                self.stride, self.pad,
                initialW=initialW, initial_bias=initial_bias,
                activ=activ)
        elif self.args_style == 'None':
            self.l = Conv2DActiv(
                None, self.out_channels, self.ksize, self.stride, self.pad,
                initialW=initialW, initial_bias=initial_bias,
                activ=activ)
        elif self.args_style == 'omit':
            self.l = Conv2DActiv(
                self.out_channels, self.ksize, stride=self.stride,
                pad=self.pad, initialW=initialW, initial_bias=initial_bias,
                activ=activ)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.l(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, self.l.xp.ndarray)

        if self.activ == 'relu':
            np.testing.assert_almost_equal(
                cuda.to_cpu(y.data), np.maximum(cuda.to_cpu(x_data), 0))
        elif self.activ == 'add_one':
            np.testing.assert_almost_equal(
                cuda.to_cpu(y.data), cuda.to_cpu(x_data) + 1)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.l.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.l(x)
        y.grad = y_grad
        y.backward()

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
