import unittest

import numpy as np

from chainer import testing
from chainer.initializers import Zero
from chainer.testing import attr
from chainer.variable import Variable

from chainercv.links import VGG16Layers


_zero_init = Zero


@testing.parameterize(
    {'feature': 'prob', 'shape': (1, 200), 'n_class': 200},
    {'feature': 'pool5', 'shape': (1, 512, 7, 7), 'n_class': None},
)
@attr.slow
class TestVGG16LayersCall(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(
            pretrained_model=None, n_class=self.n_class, feature=self.feature)

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
    {'feature': 'prob', 'shape': (2, 1000), 'do_ten_crop': False},
    {'feature': 'prob', 'shape': (2, 1000), 'do_ten_crop': True},
    {'feature': 'conv5_3', 'shape': (2, 512, 14, 14), 'do_ten_crop': False}
)
class TestVGG16LayersPredict(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(pretrained_model=None, n_class=1000,
                                feature=self.feature,
                                do_ten_crop=self.do_ten_crop)

    def check_predict(self):
        x1 = np.random.uniform(0, 255, (3, 320, 240)).astype(np.float32)
        x2 = np.random.uniform(0, 255, (3, 320, 240)).astype(np.float32)
        out = self.link.predict([x1, x2])
        self.assertEqual(out.shape, self.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()


class TestVGG16LayersCopy(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(pretrained_model=None, n_class=200,
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


class TestVGG16LayersFeatureOption(unittest.TestCase):

    def setUp(self):
        self.link = VGG16Layers(pretrained_model=None, feature='pool4',
                                initialW=Zero(), initial_bias=Zero())

    def check_feature_option(self):
        self.assertTrue(self.link.conv5_1 is None)
        self.assertTrue(self.link.conv5_2 is None)
        self.assertTrue(self.link.conv5_3 is None)
        self.assertTrue(self.link.fc6 is None)
        self.assertTrue(self.link.fc7 is None)
        self.assertTrue(self.link.fc8 is None)

        self.assertFalse('conv5_1' in self.link.functions)
        self.assertFalse('conv5_2' in self.link.functions)
        self.assertFalse('conv5_3' in self.link.functions)
        self.assertFalse('pool5' in self.link.functions)
        self.assertFalse('fc6' in self.link.functions)
        self.assertFalse('fc7' in self.link.functions)
        self.assertFalse('fc8' in self.link.functions)
        self.assertFalse('prob' in self.link.functions)
        self.assertFalse('fc8' in self.link.functions)

    def test_feature_option_cpu(self):
        self.check_feature_option()

    @attr.gpu
    def test_feature_option_gpu(self):
        self.link.to_gpu()
        self.check_feature_option()


testing.run_module(__name__, __file__)
