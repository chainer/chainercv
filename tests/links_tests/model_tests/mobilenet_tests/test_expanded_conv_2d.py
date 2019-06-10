import unittest

import numpy as np

import chainer
from chainer.backends import cuda
from chainer import testing
from chainermn import create_communicator

from chainercv.links.model.mobilenet import ExpandedConv2D
from chainercv.utils.testing import attr


@testing.parameterize(*testing.product({
    'expansion_size': [1, 2, 3],
}))
class TestExpandedConv2D(unittest.TestCase):
    in_channels = 1
    out_channels = 1
    expand_pad = 'SAME'
    depthwise_ksize = 3
    depthwise_pad = 'SAME'
    depthwise_stride = 1
    project_pad = 'SAME'

    def setUp(self):
        self.x = np.random.uniform(
            -1, 1, (5, self.in_channels, 5, 5)).astype(np.float32)
        self.gy = np.random.uniform(
            -1, 1, (5, self.out_channels, 5, 5)).astype(np.float32)

        # Convolution is the identity function.
        expand_initialW = np.ones((
            self.expansion_size, self.in_channels),
            dtype=np.float32).reshape(
                (self.expansion_size, self.in_channels, 1, 1))
        depthwise_initialW = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]*self.expansion_size,
                                      dtype=np.float32).reshape((self.expansion_size, 1, 3, 3))
        project_initialW = np.ones(
            (self.out_channels, self.expansion_size),
            dtype=np.float32).reshape(
                (self.out_channels, self.expansion_size, 1, 1))
        bn_kwargs = {'decay': 0.8}
        self.l = ExpandedConv2D(
            self.in_channels, self.out_channels, expansion_size=self.expansion_size,
            expand_pad=self.expand_pad, depthwise_stride=self.depthwise_stride,
            depthwise_ksize=self.depthwise_ksize, depthwise_pad=self.depthwise_pad,
            project_pad=self.project_pad, bn_kwargs=bn_kwargs)
        if self.expansion_size > self.in_channels:
            self.l.expand.conv.W.array = expand_initialW
        self.l.depthwise.conv.W.array = depthwise_initialW
        self.l.project.conv.W.array = project_initialW

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        # Make the batch normalization to be the identity function.
        if self.expansion_size != 1:
            self.l.expand.bn.avg_var[:] = 1
            self.l.expand.bn.avg_mean[:] = 0
        self.l.depthwise.bn.avg_var[:] = 1
        self.l.depthwise.bn.avg_mean[:] = 0
        self.l.project.bn.avg_var[:] = 1
        self.l.project.bn.avg_mean[:] = 0
        with chainer.using_config('train', False):
            y = self.l(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.array, self.l.xp.ndarray)

        _x_data = x_data
        if self.expansion_size > self.in_channels:
            np.testing.assert_almost_equal(
                cuda.to_cpu(y.array), _x_data+self.expansion_size *
                np.maximum(np.minimum(cuda.to_cpu(_x_data), 6), 0),
                decimal=4
            )
        else:
            np.testing.assert_almost_equal(
                cuda.to_cpu(y.array), _x_data +
                np.maximum(np.minimum(cuda.to_cpu(_x_data), 6), 0),
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


testing.run_module(__name__, __file__)
