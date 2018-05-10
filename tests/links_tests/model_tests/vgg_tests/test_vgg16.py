import unittest

import numpy as np

import chainer
from chainer.initializers import Zero
from chainer import testing
from chainer import Variable

from chainercv.links import VGG16
from chainercv.testing import attr


@testing.parameterize(
    {'pick': 'prob', 'shapes': (1, 200), 'n_class': 200},
    {'pick': 'pool5', 'shapes': (1, 512, 7, 7), 'n_class': None},
    {'pick': ['conv5_3', 'conv4_2'],
     'shapes': ((1, 512, 14, 14), (1, 512, 28, 28)), 'n_class': None}
)
class TestVGG16Call(unittest.TestCase):

    def setUp(self):
        self.link = VGG16(
            n_class=self.n_class, pretrained_model=None,
            initialW=Zero())
        self.link.pick = self.pick

    def check_call(self):
        xp = self.link.xp

        x1 = Variable(xp.asarray(np.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(np.float32)))
        with chainer.no_backprop_mode():
            features = self.link(x1)
        if isinstance(features, tuple):
            for activation, shape in zip(features, self.shapes):
                self.assertEqual(activation.shape, shape)
        else:
            self.assertEqual(features.shape, self.shapes)
            self.assertEqual(features.dtype, np.float32)

    @attr.slow
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


class TestVGG16Pretrained(unittest.TestCase):

    @attr.disk
    @attr.slow
    def test_pretrained(self):
        VGG16(pretrained_model='imagenet')

    @attr.disk
    @attr.slow
    def test_pretrained_n_class(self):
        VGG16(n_class=1000, pretrained_model='imagenet')

    @attr.disk
    @attr.slow
    def test_pretrained_wrong_n_class(self):
        with self.assertRaises(ValueError):
            VGG16(n_class=100, pretrained_model='imagenet')


testing.run_module(__name__, __file__)
