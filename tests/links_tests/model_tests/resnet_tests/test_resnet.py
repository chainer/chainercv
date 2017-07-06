import unittest

import numpy as np

from chainer.initializers import Zero
from chainer import testing
from chainer.testing import attr
from chainer import Variable

from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50


@testing.parameterize(*(
    testing.product_dict(
        [
            {'layer_names': 'prob', 'shapes': (1, 200), 'n_class': 200},
            {'layer_names': 'res5',
             'shapes': (1, 2048, 7, 7), 'n_class': None},
            {'layer_names': ['res2', 'conv1'],
             'shapes': ((1, 256, 56, 56), (1, 64, 112, 112)), 'n_class': None},
        ],
        [
            {'model_class': ResNet50},
            {'model_class': ResNet101},
            {'model_class': ResNet152},
        ]
    )
))
@attr.slow
class TestResNetCall(unittest.TestCase):

    def setUp(self):
        self.link = self.model_class(
            pretrained_model=None, n_class=self.n_class,
            layer_names=self.layer_names)

    def check_call(self):
        xp = self.link.xp

        x1 = Variable(xp.asarray(np.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(np.float32)))
        features = self.link(x1)
        if isinstance(features, tuple):
            for activation, shape in zip(features, self.shapes):
                self.assertEqual(activation.shape, shape)
        else:
            self.assertEqual(features.shape, self.shapes)
            self.assertEqual(features.dtype, np.float32)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


@testing.parameterize(
    {'model_class': ResNet50},
    {'model_class': ResNet101},
    {'model_class': ResNet152}
)
class TestResNetCopy(unittest.TestCase):

    def setUp(self):
        self.link = self.model_class(
            pretrained_model=None, n_class=200,
            layer_names='res2', initialW=Zero())

    def check_copy(self):
        copied = self.link.copy()
        self.assertIs(copied.conv1, copied.layers['conv1'])

    def test_copy_cpu(self):
        self.check_copy()

    @attr.gpu
    def test_copy_gpu(self):
        self.link.to_gpu()
        self.check_copy()


@testing.parameterize(*(
    testing.product_dict(
        [
            {'layer_names': 'res4', 'not_attribute': ['res5', 'fc6']},
            {'layer_names': ['res5', 'res2'], 'not_attribute': ['fc6']}
        ],
        [
            {'model_class': ResNet50},
            {'model_class': ResNet101},
            {'model_class': ResNet152},
        ]
    )
))
class TestResNetFeatureOption(unittest.TestCase):

    def setUp(self):
        self.link = ResNet50(
            pretrained_model=None, layer_names=self.layer_names,
            initialW=Zero())

    def check_feature_option(self):
        for name in self.not_attribute:
            self.assertTrue(not hasattr(self.link, name))

    def test_feature_option_cpu(self):
        self.check_feature_option()

    @attr.gpu
    def test_feature_option_gpu(self):
        self.link.to_gpu()
        self.check_feature_option()


testing.run_module(__name__, __file__)
