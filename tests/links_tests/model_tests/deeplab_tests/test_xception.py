import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.deeplab.xception import Xception65


class TestXception(unittest.TestCase):

    def setUp(self):
        self.link = Xception65()

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.random.uniform(
            low=-1, high=1, size=(2, 3, 64, 64)
        ).astype(xp.float32))
        y1, y2 = self.link(x)

        self.assertIsInstance(y1, chainer.Variable)
        self.assertIsInstance(y1.data, xp.ndarray)
        self.assertEqual(y1.shape, (2, 256, 16, 16))
        self.assertIsInstance(y2, chainer.Variable)
        self.assertIsInstance(y2.data, xp.ndarray)
        self.assertEqual(y2.shape, (2, 2048, 8, 8))

    @attr.slow
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
