import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import SegNetBasic


@testing.parameterize(
    {'train': False},
    {'train': True}
)
@attr.slow
class TestFasterRCNNVGG16(unittest.TestCase):
    
    def setUp(self):
        self.n_class = 10
        self.link = SegNetBasic(n_class=self.n_class)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.random.uniform(
            low=-1, high=1, size=(2, 3, 224, 256)).astype(np.float32))
        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, xp.ndarray)
        self.assertEqual(y.shape, (2, self.n_class, 224, 256))

    def test_call_cpu(self):
        self.check_call()

    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
