import numpy as np
import unittest
from unittest.mock import MagicMock

import chainer
from chainer import testing
from chainer.testing import attr
from chainercv.links import PixelwiseSigmoidClassifier
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.links import PixelwiseSoftmaxWithWeightClassifier
from chainercv.links.loss import semantic_segmentation_loss as ssloss

class DummyCNN(chainer.Chain):

    def __init__(self, n_class):
        super(DummyCNN, self).__init__()
        self.n_class = n_class

    def __call__(self, x):
        n, _, h, w = x.shape
        y = self.xp.random.rand(n, self.n_class, h, w).astype(np.float32)
        return chainer.Variable(y)
        

@testing.parameterize(
    {'n_class': 11, 'compute_accuracy': True},
    {'n_class': 11, 'compute_accuracy': False},
)
class TestPixelwiseSigmoidClassifier(unittest.TestCase):

    def setUp(self):
        model = DummyCNN(self.n_class)
        self.link = PixelwiseSigmoidClassifier(
            model, self.n_class, self.compute_accuracy)
        self.x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        self.t = np.random.randint(
                    self.n_class, size=(1, 11, 32, 32)).astype(np.int32)
        ssloss._segmentation_accuracies = MagicMock()

    def _check_call(self):
        xp = self.link.xp
        loss = self.link(xp.asarray(self.x), xp.asarray(self.t))
        self.assertIsInstance(loss, chainer.Variable)
        self.assertIsInstance(loss.data, self.link.xp.ndarray)
        self.assertEqual(loss.shape, ())
        
        if self.compute_accuracy:
            ssloss._segmentation_accuracies.assert_called_with(
                self.link.y, self.t, self.n_class)
            print('aaa')


    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def test_call_cpu(self):
        self._check_call()
        
