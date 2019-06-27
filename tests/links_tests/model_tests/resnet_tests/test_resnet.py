import unittest

import numpy as np

from chainer.testing import attr
from chainer import Variable

from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50
from chainercv.utils import testing


@testing.parameterize(*(
    testing.product_dict(
        [
            {'pick': 'prob', 'shapes': (1, 200), 'n_class': 200},
            {'pick': 'res5',
             'shapes': (1, 2048, 7, 7), 'n_class': None},
            {'pick': ['res2', 'conv1'],
             'shapes': ((1, 256, 56, 56), (1, 64, 112, 112)), 'n_class': None},
        ],
        [
            {'model_class': ResNet50},
            {'model_class': ResNet101},
            {'model_class': ResNet152},
        ],
        [
            {'arch': 'fb'},
            {'arch': 'he'}
        ],
        [
            {'weight_standardization': False},
            {'weight_standardization': True}
        ],
        [
            {'norm_type': 'bn'},
            {'norm_type': 'gn'}
        ]
    )
))
class TestResNetCall(unittest.TestCase):

    def setUp(self):
        self.link = self.model_class(
            n_class=self.n_class, pretrained_model=None, arch=self.arch,
            weight_standardization=self.weight_standardization,
            norm_type=self.norm_type
            )
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
    'model': [ResNet50, ResNet101, ResNet152],
    'n_class': [None, 500, 1000],
    'pretrained_model': ['imagenet'],
    'mean': [None, np.random.uniform((3, 1, 1)).astype(np.float32)],
    'arch': ['he', 'fb'],
}))
class TestResNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_class': self.n_class,
            'pretrained_model': self.pretrained_model,
            'mean': self.mean,
            'arch': self.arch,
        }

        if self.pretrained_model == 'imagenet':
            valid = self.n_class in {None, 1000}

        if valid:
            self.model(**kwargs)
        else:
            with self.assertRaises(ValueError):
                self.model(**kwargs)


testing.run_module(__name__, __file__)
