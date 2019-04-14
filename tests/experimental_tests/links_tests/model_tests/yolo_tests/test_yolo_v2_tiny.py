import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links import YOLOv2Tiny


@testing.parameterize(*testing.product({
    'n_fg_class': [1, 5, 20],
}))
class TestYOLOv2Tiny(unittest.TestCase):

    def setUp(self):
        self.link = YOLOv2Tiny(n_fg_class=self.n_fg_class)
        self.insize = 416
        self.n_bbox = 13 * 13 * 5

    def _check_call(self):
        x = self.link.xp.array(
            np.random.uniform(-1, 1, size=(1, 3, self.insize, self.insize)),
            dtype=np.float32)

        locs, objs, confs = self.link(x)

        self.assertIsInstance(locs, chainer.Variable)
        self.assertIsInstance(locs.array, self.link.xp.ndarray)
        self.assertEqual(locs.shape, (1, self.n_bbox, 4))

        self.assertIsInstance(objs, chainer.Variable)
        self.assertIsInstance(objs.array, self.link.xp.ndarray)
        self.assertEqual(objs.shape, (1, self.n_bbox))

        self.assertIsInstance(confs, chainer.Variable)
        self.assertIsInstance(confs.array, self.link.xp.ndarray)
        self.assertEqual(confs.shape, (1, self.n_bbox, self.n_fg_class))

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
class TestYOLOv2TinyPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_fg_class': self.n_fg_class,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model == 'voc0712':
            valid = self.n_fg_class in {None, 20}

        if valid:
            YOLOv2Tiny(**kwargs)
        else:
            with self.assertRaises(ValueError):
                YOLOv2Tiny(**kwargs)


testing.run_module(__name__, __file__)
