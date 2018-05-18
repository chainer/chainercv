import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import YOLOv2


@testing.parameterize(*testing.product({
    'n_fg_class': [1, 5, 20],
}))
class TestYOLOv2(unittest.TestCase):

    def setUp(self):
        self.link = YOLOv2(n_fg_class=self.n_fg_class)
        self.insize = 416
        self.n_bbox = 13 * 13 * 5

    def _check_call(self):
        x = self.link.xp.array(
            np.random.uniform(-1, 1, size=(1, 3, self.insize, self.insize)),
            dtype=np.float32)

        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.array, self.link.xp.ndarray)
        self.assertEqual(y.shape, (1, self.n_bbox, 4 + 1 + self.n_fg_class))

    @attr.slow
    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()


@testing.parameterize(*testing.product({
    'n_fg_class': [None, 10, 20],
    'pretrained_model': ['voc0712'],
}))
class TestYOLOv2Pretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_fg_class': self.n_fg_class,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model == 'voc0712':
            valid = self.n_fg_class in {None, 20}

        if valid:
            YOLOv2(**kwargs)
        else:
            with self.assertRaises(ValueError):
                YOLOv2(**kwargs)


testing.run_module(__name__, __file__)
