import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import FeatureExtractionPredictor


class DummyFeatureExtractor(chainer.Chain):

    mean = np.array([0, 0, 0]).reshape(3, 1, 1)

    def __init__(self, shape_0, shape_1):
        super(DummyFeatureExtractor, self).__init__()
        self.shape_0 = shape_0
        self.shape_1 = shape_1

    def __call__(self, x):
        shape = (x.shape[0],) + self.shape_0
        y0 = self.xp.random.rand(*shape).astype(np.float32)

        if self.shape_1 is None:
            return chainer.Variable(y0)
        shape = (x.shape[0],) + self.shape_1
        y1 = self.xp.random.rand(*shape).astype(np.float32)
        return chainer.Variable(y0), chainer.Variable(y1)


@testing.parameterize(
    {'shape_0': (5, 10, 10), 'shape_1': None, 'crop': 'center'},
    {'shape_0': (8,), 'shape_1': None, 'crop': '10'},
    {'shape_0': (5, 10, 10), 'shape_1': (12,), 'crop': 'center'},
    {'shape_0': (8,), 'shape_1': (10,), 'crop': '10'},
)
class TestFeatureExtractionPredictorPredict(unittest.TestCase):

    def setUp(self):
        self.link = FeatureExtractionPredictor(
            DummyFeatureExtractor(self.shape_0, self.shape_1),
            crop=self.crop)
        self.x = np.random.uniform(size=(3, 3, 32, 32)).astype(np.float32)

        self.one_output = self.shape_1 is None

    def check(self, x):
        out = self.link.predict(x)
        if self.one_output:
            self.assertEqual(out.shape, (self.x.shape[0],) + self.shape_0)
            self.assertIsInstance(out, np.ndarray)
        else:
            out_0, out_1 = out
            self.assertEqual(out_0.shape, (self.x.shape[0],) + self.shape_0)
            self.assertEqual(out_1.shape, (self.x.shape[0],) + self.shape_1)
            self.assertIsInstance(out_0, np.ndarray)
            self.assertIsInstance(out_1, np.ndarray)

    def test_cpu(self):
        self.check(self.x)

    @attr.gpu
    def test_gpu(self):
        self.link.to_gpu()
        self.check(self.x)


@testing.parameterize(
    {'crop': 'center', 'crop_size': 192},
    {'crop': '10', 'crop_size': 192}
)
class TestFeatureExtractionPredictorPrepare(unittest.TestCase):

    n_channel = 3

    def setUp(self):
        self.link = FeatureExtractionPredictor(
            DummyFeatureExtractor((1,), None),
            crop_size=self.crop_size, crop=self.crop)
        if self.crop == 'center':
            self.expected_shape = (
                1, self.n_channel, self.crop_size, self.crop_size)
        elif self.crop == '10':
            self.expected_shape = (
                10, self.n_channel, self.crop_size, self.crop_size)

    def test(self):
        out = self.link._prepare(
            np.random.uniform(size=(self.n_channel, 128, 256)))

        self.assertEqual(out.shape, self.expected_shape)


testing.run_module(__name__, __file__)
