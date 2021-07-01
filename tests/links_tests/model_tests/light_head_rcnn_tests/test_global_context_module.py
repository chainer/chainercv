import numpy as np
import unittest

import chainer
from chainer.testing import attr


from chainercv.links.model.light_head_rcnn.global_context_module \
    import GlobalContextModule


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class TestGlobalContextModule(unittest.TestCase):

    def setUp(self):
        self.in_channels = 4
        self.mid_channels = 4
        self.out_channels = 4
        self.ksize = 7
        self.H = 24
        self.W = 32
        self.global_context_module = GlobalContextModule(
            self.in_channels, self.mid_channels,
            self.out_channels, self.ksize)

    def check_call(self):
        xp = self.global_context_module.xp
        x = chainer.Variable(
            _random_array(xp, (1, self.in_channels, self.H, self.W)))
        y = self.global_context_module(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.array, xp.ndarray)
        self.assertEqual(y.shape, (1, self.out_channels, self.H, self.W))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.global_context_module.to_gpu()
        self.check_call()
