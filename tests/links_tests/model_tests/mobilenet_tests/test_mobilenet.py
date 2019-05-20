import unittest

import numpy as np

from chainer.testing import attr
from chainer import Variable

from chainercv.links import MobileNetV2
from chainercv.utils import testing


@testing.parameterize(*(
    testing.product_dict(
        [
            {'pick': 'softmax', 'shapes': (1, 200), 'n_class': 200},
            {'pick': 'conv1',
             'shapes': (1, 1280, 7, 7), 'n_class': None},
            {'pick': ['expanded_conv_2', 'conv'],
             'shapes': ((1, 24, 56, 56), (1, 32, 112, 112)), 'n_class': None},
        ],
        [
            {'model_class': MobileNetV2},
        ],
        [
            {'arch': 'tf'},
        ]
    )
))
class TestMobileNetCall(unittest.TestCase):

    def setUp(self):
        self.link = self.model_class(
            n_class=self.n_class, pretrained_model=None, arch=self.arch)
        self.link.pick = self.pick

    def check_call(self):
        xp = self.link.xp

        x = Variable(xp.asarray(np.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(np.float32)))
        features = self.link(x)
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


@testing.parameterize(*testing.product({
    'model': [MobileNetV2],
    'n_class': [None, 500, 1001],
    'pretrained_model': ['imagenet'],
    'mean': [None, np.random.uniform((3, 1, 1)).astype(np.float32)],
    'scale': [None, np.random.uniform((3, 1, 1)).astype(np.float32)],
    'arch': ['tf'],
}))
class TestMobileNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_class': self.n_class,
            'pretrained_model': self.pretrained_model,
            'mean': self.mean,
            'scale': self.scale,
            'arch': self.arch,
        }

        if self.pretrained_model == 'imagenet':
            valid = self.n_class in {None, 1001}

        if valid:
            self.model(**kwargs)
        else:
            with self.assertRaises(ValueError):
                self.model(**kwargs)


testing.run_module(__name__, __file__)
