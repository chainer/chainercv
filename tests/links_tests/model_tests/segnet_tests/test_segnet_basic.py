import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import SegNetBasic
from chainercv.utils import assert_is_semantic_segmentation_link


@testing.parameterize(
    {'train': False},
    {'train': True}
)
class TestSegNetBasic(unittest.TestCase):

    def setUp(self):
        self.n_class = 10
        self.link = SegNetBasic(n_class=self.n_class)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.random.uniform(
            low=-1, high=1, size=(2, 3, 128, 160)).astype(np.float32))
        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.array, xp.ndarray)
        self.assertEqual(y.shape, (2, self.n_class, 128, 160))

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
    'n_class': [None, 5, 11],
    'pretrained_model': ['camvid'],
}))
class TestSegNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_class': self.n_class,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model == 'camvid':
            valid = self.n_class in {None, 11}

        if valid:
            SegNetBasic(**kwargs)
        else:
            with self.assertRaises(ValueError):
                SegNetBasic(**kwargs)


testing.run_module(__name__, __file__)
