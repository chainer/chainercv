import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr
from chainercv.links import PixelwiseSoftmaxClassifier


class DummySemanticSegmentationModel(chainer.Chain):

    def __init__(self, n_class):
        super(DummySemanticSegmentationModel, self).__init__()
        self.n_class = n_class

    def __call__(self, x):
        n, _, h, w = x.shape
        y = self.xp.random.rand(n, self.n_class, h, w).astype(np.float32)
        return chainer.Variable(y)


@testing.parameterize(
    {'n_class': 11, 'ignore_label': -1, 'class_weight': True},
    {'n_class': 11, 'ignore_label': 11, 'class_weight': None},
)
class TestPixelwiseSoftmaxClassifier(unittest.TestCase):

    def setUp(self):
        model = DummySemanticSegmentationModel(self.n_class)
        if self.class_weight:
            self.class_weight = [0.1 * i for i in range(self.n_class)]
        self.link = PixelwiseSoftmaxClassifier(
            model, self.ignore_label, self.class_weight)
        self.x = np.random.rand(2, 3, 16, 16).astype(np.float32)
        self.t = np.random.randint(
            self.n_class, size=(2, 16, 16)).astype(np.int32)

    def _check_call(self):
        xp = self.link.xp
        loss = self.link(chainer.Variable(xp.asarray(self.x)),
                         chainer.Variable(xp.asarray(self.t)))
        self.assertIsInstance(loss, chainer.Variable)
        self.assertIsInstance(loss.data, self.link.xp.ndarray)
        self.assertEqual(loss.shape, ())

        self.assertTrue(hasattr(self.link, 'y'))
        self.assertIsNotNone(self.link.y)

        self.assertTrue(hasattr(self.link, 'loss'))
        xp.testing.assert_allclose(self.link.loss.data, loss.data)

    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()
