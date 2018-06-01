import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links import PSPNetResNet101
from chainercv.utils import assert_is_semantic_segmentation_link


class TestPSPNetResNet101(unittest.TestCase):

    def setUp(self):
        self.n_class = 10
        self.input_size = (120, 160)
        self.link = PSPNetResNet101(
            n_class=self.n_class, input_size=self.input_size)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.random.uniform(
            low=-1, high=1, size=(2, 3) + self.input_size).astype(np.float32))
        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, xp.ndarray)
        self.assertEqual(y.shape, (2, self.n_class, 120, 160))

    @attr.slow
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    @attr.slow
    def test_predict_cpu(self):
        assert_is_semantic_segmentation_link(self.link, self.n_class)

    @attr.gpu
    @attr.slow
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_semantic_segmentation_link(self.link, self.n_class)


@testing.parameterize(*testing.product({
    'n_class': [None, 5, 19],
    'pretrained_model': ['cityscapes'],
}))
class TestPSPNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_class': self.n_class,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model == 'cityscapes':
            valid = self.n_class in {None, 19}

        if valid:
            PSPNetResNet101(**kwargs)
        else:
            with self.assertRaises(ValueError):
                PSPNetResNet101(**kwargs)


testing.run_module(__name__, __file__)
