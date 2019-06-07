import unittest

import numpy as np

import chainer
from chainer.backends import cuda
from chainer import testing
from chainermn import create_communicator

from chainercv.links.model.mobilenet import TFConvolution2D
from chainercv.utils.testing import attr


def _add_one(x):
    return x + 1


@testing.parameterize(*testing.product({
    'pad': [1, 'SAME'],
    'args_style': ['explicit', 'None', 'omit'],
}))
class TestTFConvolution2D(unittest.TestCase):

    in_channels = 1
    out_channels = 1
    ksize = 3
    stride = 1
    pad = 1
    dilate = 1

    def setUp(self):
        self.x = np.random.uniform(
            -1, 1, (5, self.in_channels, 5, 5)).astype(np.float32)
        self.gy = np.random.uniform(
            -1, 1, (5, self.out_channels, 5, 5)).astype(np.float32)

        # Convolution is the identity function.
        initialW = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                            dtype=np.float32).reshape((1, 1, 3, 3))
        initial_bias = 0
        if self.args_style == 'explicit':
            self.l = TFConvolution2D(
                self.in_channels, self.out_channels, self.ksize,
                self.stride, self.pad, self.dilate,
                initialW=initialW, initial_bias=initial_bias)
        elif self.args_style == 'None':
            self.l = TFConvolution2D(
                None, self.out_channels, self.ksize, self.stride, self.pad,
                self.dilate, initialW=initialW, initial_bias=initial_bias)
        elif self.args_style == 'omit':
            self.l = TFConvolution2D(
                self.out_channels, self.ksize, stride=self.stride,
                pad=self.pad, dilate=self.dilate, initialW=initialW,
                initial_bias=initial_bias)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with chainer.using_config('train', False):
            y = self.l(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.array, self.l.xp.ndarray)

        _x_data = x_data
        np.testing.assert_almost_equal(
            cuda.to_cpu(y.array), cuda.to_cpu(_x_data),
            decimal=4
        )

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


@attr.mpi
class TestTFConv2DMultiNodeBNActiv(unittest.TestCase):

    in_channels = 1
    out_channels = 1
    ksize = 3
    stride = 1
    pad = 1
    dilate = 1

    def setUp(self):
        self.x = np.random.uniform(
            -1, 1, (5, self.in_channels, 5, 5)).astype(np.float32)
        self.gy = np.random.uniform(
            -1, 1, (5, self.out_channels, 5, 5)).astype(np.float32)

        # Convolution is the identity function.
        initialW = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                            dtype=np.float32).reshape((1, 1, 3, 3))
        initial_bias = 0
        activ = relu
        self.l = TFConvolution2D(
            self.in_channels, self.out_channels, self.ksize, self.stride,
            self.pad, self.dilate, initialW=initialW,
            initial_bias=initial_bias)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with chainer.using_config('train', False):
            y = self.l(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.array, self.l.xp.ndarray)

        np.testing.assert_almost_equal(
            cuda.to_cpu(y.array), np.maximum(cuda.to_cpu(x_data), 0),
            decimal=4
        )

    def test_multi_node_batch_normalization_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_multi_node_batch_normalization_forward_gpu(self):
        self.l.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.l(x)
        y.grad = y_grad
        y.backward()

    def test_multi_node_batch_normalization_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_multi_node_batch_normalization_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
