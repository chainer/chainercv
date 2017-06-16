import unittest

import numpy as np

from chainer.initializers import Zero
from chainer import testing
from chainer.testing import attr
from chainer import Variable

from chainercv.links import VGG16Layers


@testing.parameterize(
    {'features': 'prob', 'shape': (1, 200), 'n_class': 200},
    {'features': 'pool5', 'shape': (1, 512, 7, 7), 'n_class': None},
)
@attr.slow
class TestVGG16LayersCall(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(
            pretrained_model=None, n_class=self.n_class, feature=self.features)

    def check_call(self):
        xp = self.link.xp

        x1 = Variable(xp.asarray(np.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(np.float32)))
        y1 = self.link(x1)
        self.assertEqual(y1.shape, self.shape)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


@testing.parameterize(
    {'features': 'prob', 'shape': (2, 1000), 'do_ten_crop': False},
    {'features': 'prob', 'shape': (2, 1000), 'do_ten_crop': True},
    {'features': 'conv5_3', 'shape': (2, 512, 14, 14), 'do_ten_crop': False},
    {'features': ['fc6', 'conv3_1'],
     'shape': {'conv3_1': (2, 256, 56, 56), 'fc6': (2, 4096)},
     'do_ten_crop': False},
    {'features': ['fc6', 'fc7'],
     'shape': {'fc6': (2, 4096), 'fc7': (2, 4096)}, 'do_ten_crop': True}
)
@attr.slow
class TestVGG16LayersPredict(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(pretrained_model=None, n_class=1000,
                                features=self.features,
                                do_ten_crop=self.do_ten_crop)

    def check_predict(self):
        x1 = np.random.uniform(0, 255, (3, 320, 240)).astype(np.float32)
        x2 = np.random.uniform(0, 255, (3, 320, 240)).astype(np.float32)
        activations = self.link.predict([x1, x2])
        if isinstance(activations, dict):
            for name in self.features:
                self.assertEqual(activations[name].shape, self.shape[name])
                self.assertEqual(activations[name].dtype, np.float32)
        else:
            self.assertEqual(activations.shape, self.shape)
            self.assertEqual(activations.dtype, np.float32)

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()


class TestVGG16LayersCopy(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(pretrained_model=None, n_class=200,
                                features='conv2_2',
                                initialW=Zero(), initial_bias=Zero())

    def check_copy(self):
        copied = self.link.copy()
        self.assertIs(copied.conv1_1, copied.functions['conv1_1'][0])

    def test_copy_cpu(self):
        self.check_copy()

    @attr.gpu
    def test_copy_gpu(self):
        self.link.to_gpu()
        self.check_copy()


@testing.parameterize(
    {'features': 'pool4',
     'not_attribute': ['conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8'],
     'not_in_functions': ['conv5_1', 'conv5_2', 'conv5_3', 'pool5',
                          'fc6', 'fc7', 'fc8', 'prob']
     },
    {'features': ['pool5', 'pool4'],
     'not_attribute': ['fc6', 'fc7', 'fc8'],
     'not_in_functions': ['fc6', 'fc7', 'fc8', 'prob']
     }
)
class TestVGG16LayersFeatureOption(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(pretrained_model=None, features=self.features,
                                initialW=Zero(), initial_bias=Zero())

    def check_feature_option(self):
        for name in self.not_attribute:
            self.assertTrue(not hasattr(self.link, name))

        for name in self.not_in_functions:
            self.assertFalse(name in self.link.functions)

    def test_feature_option_cpu(self):
        self.check_feature_option()

    @attr.gpu
    def test_feature_option_gpu(self):
        self.link.to_gpu()
        self.check_feature_option()


testing.run_module(__name__, __file__)
