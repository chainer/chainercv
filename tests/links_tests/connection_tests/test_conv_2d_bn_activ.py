import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links import Conv2DBNActiv


@testing.parameterize(
    {'args_style': 'explicit'},
    {'args_style': 'None'},
    {'args_style': 'omit'}
)
class TestConv2DBNActiv(unittest.TestCase):

    in_channels = 3
    out_channels = 5
    ksize = 3
    stride = 1
    pad = 1

    def setUp(self):
        self.x = np.random.uniform(
            -1, 1, (5, self.in_channels, 5, 5)).astype(np.float32)
        self.gy = np.random.uniform(
            -1, 1, (5, self.out_channels, 5, 5)).astype(np.float32)
        if self.args_style == 'explicit':
            self.l = Conv2DBNActiv(
                self.in_channels, self.out_channels, self.ksize,
                self.stride, self.pad)
        elif self.args_style == 'None':
            self.l = Conv2DBNActiv(
                None, self.out_channels, self.ksize,
                self.stride, self.pad)
        elif self.args_style == 'omit':
            self.l = Conv2DBNActiv(
                self.out_channels, self.ksize,
                stride=self.stride, pad=self.pad)

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
