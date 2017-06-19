import unittest

import numpy as np

from chainer.initializers import Zero
from chainer import testing
from chainer.testing import attr
from chainer import Variable

from chainercv.links import VGG16


@testing.parameterize(
    {'feature_names': 'prob', 'shapes': (1, 200), 'n_class': 200},
    {'feature_names': 'pool5', 'shapes': (1, 512, 7, 7), 'n_class': None},
    {'feature_names': ['conv5_3', 'conv4_2'],
     'shapes': ((1, 512, 14, 14), (1, 512, 28, 28)), 'n_class': None},
)
@attr.slow
class TestVGG16Call(unittest.TestCase):

    def setUp(self):
        self.link = VGG16(
            pretrained_model=None, n_class=self.n_class,
            feature_names=self.feature_names)

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


class TestVGG16Copy(unittest.TestCase):

    def setUp(self):
        self.link = VGG16(pretrained_model=None, n_class=200,
                          feature_names='conv2_2',
                          initialW=Zero(), initial_bias=Zero())

    def check_copy(self):
        copied = self.link.copy()
        self.assertIs(copied.conv1_1, copied.functions['conv1_1'])

    def test_copy_cpu(self):
        self.check_copy()

    @attr.gpu
    def test_copy_gpu(self):
        self.link.to_gpu()
        self.check_copy()


@testing.parameterize(
    {'feature_names': 'pool4',
     'not_attribute': ['conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8'],
     },
    {'feature_names': ['pool5', 'pool4'],
     'not_attribute': ['fc6', 'fc7', 'fc8'],
     }
)
class TestVGG16FeatureOption(unittest.TestCase):

    def setUp(self):
        self.link = VGG16(
            pretrained_model=None, feature_names=self.feature_names,
            initialW=Zero(), initial_bias=Zero())

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
