import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.deeplab import SeparableASPP


class TestSeparableASPP(unittest.TestCase):

    def setUp(self):
        self.in_channels = 128
        self.out_channels = 32
        self.link = SeparableASPP(
            self.in_channels, self.out_channels)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.random.uniform(
            low=-1, high=1, size=(2, self.in_channels, 64, 64)
        ).astype(xp.float32))
        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, xp.ndarray)
        self.assertEqual(y.shape, (2, self.out_channels, 64, 64))

    @attr.slow
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
