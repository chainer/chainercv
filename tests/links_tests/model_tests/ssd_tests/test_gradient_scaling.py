import numpy as np
import unittest

import chainer
from chainer import optimizers
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import GradientScaling


class SimpleLink(chainer.Link):

    def __init__(self, w, g):
        super(SimpleLink, self).__init__()
        with self.init_scope():
            self.param = chainer.Parameter(w)
            self.param.grad = g


class TestGradientScaling(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape((2, 3)),
            np.arange(3, -3, -1, dtype=np.float32).reshape((2, 3)))

    def check_gradient_scaling(self):
        w = self.target.param.array
        g = self.target.param.grad

        rate = 0.2
        expect = w - g * rate

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(GradientScaling(rate))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_gradient_scaling_cpu(self):
        self.check_gradient_scaling()

    @attr.gpu
    def test_gradient_scaling_gpu(self):
        self.target.to_gpu()
        self.check_gradient_scaling()


testing.run_module(__name__, __file__)
