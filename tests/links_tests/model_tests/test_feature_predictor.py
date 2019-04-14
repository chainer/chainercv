import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import FeaturePredictor


class DummyFeatureExtractor(chainer.Chain):

    def __init__(self, in_channels, shape_0, shape_1):
        super(DummyFeatureExtractor, self).__init__()
        self.shape_0 = shape_0
        self.shape_1 = shape_1
        self.mean = np.zeros(in_channels).reshape((in_channels, 1, 1))

    def forward(self, x):
        shape = (x.shape[0],) + self.shape_0
        y0 = self.xp.random.rand(*shape).astype(np.float32)

        if self.shape_1 is None:
            return chainer.Variable(y0)
        shape = (x.shape[0],) + self.shape_1
        y1 = self.xp.random.rand(*shape).astype(np.float32)
        return chainer.Variable(y0), chainer.Variable(y1)


@testing.parameterize(*(
    testing.product_dict(
        [
            {'shape_0': (5, 10, 10), 'shape_1': None, 'crop': 'center'},
            {'shape_0': (8,), 'shape_1': None, 'crop': '10'},
            {'shape_0': (5, 10, 10), 'shape_1': (12,), 'crop': 'center'},
            {'shape_0': (8,), 'shape_1': (10,), 'crop': '10'}
        ],
        [
            {'in_channels': 1},
            {'in_channels': 3}
        ]
    )
))
class TestFeaturePredictorPredict(unittest.TestCase):

    def setUp(self):
        self.link = FeaturePredictor(
            DummyFeatureExtractor(
                self.in_channels, self.shape_0, self.shape_1),
            crop_size=5, crop=self.crop)
        self.x = np.random.uniform(
            size=(3, self.in_channels, 32, 32)).astype(np.float32)

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


@testing.parameterize(*testing.product({
    'crop': ['center', '10'],
    'crop_size': [192, (192, 256), (256, 192)],
    'scale_size': [None, 256, (256, 256)],
    'in_channels': [1, 3],
    'mean': [None, np.float32(1)]
}))
class TestFeaturePredictor(unittest.TestCase):

    def setUp(self):

        self.link = FeaturePredictor(
            DummyFeatureExtractor(self.in_channels, (1,), None),
            crop_size=self.crop_size, scale_size=self.scale_size,
            crop=self.crop, mean=self.mean)

        if isinstance(self.crop_size, int):
            hw = (self.crop_size, self.crop_size)
        else:
            hw = self.crop_size
        if self.crop == 'center':
            self.expected_shape = (1, self.in_channels) + hw
        elif self.crop == '10':
            self.expected_shape = (10, self.in_channels) + hw

    def test_prepare(self):
        out = self.link._prepare(
            np.random.uniform(size=(self.in_channels, 286, 286)))

        self.assertEqual(out.shape, self.expected_shape)

    def test_prepare_original_unaffected(self):
        original = np.random.uniform(size=(self.in_channels, 286, 286))
        input_ = original.copy()
        self.link._prepare(input_)
        np.testing.assert_equal(original, input_)

    def test_mean(self):
        if self.mean is None:
            np.testing.assert_equal(self.link.mean, self.link.extractor.mean)
        else:
            np.testing.assert_equal(self.link.mean, self.mean)


testing.run_module(__name__, __file__)
