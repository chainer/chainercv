import copy
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
        params = copy.deepcopy(YOLOv2Tiny.preset_params['voc'])
        params['n_fg_class'] = self.n_fg_class
        self.link = YOLOv2Tiny(**params)
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
    'n_fg_class': [10, 20],
    'pretrained_model': ['voc0712'],
}))
class TestYOLOv2TinyPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        params = copy.deepcopy(YOLOv2Tiny.preset_params['voc'])
        params['n_fg_class'] = self.n_fg_class

        if self.pretrained_model == 'voc0712':
            valid = self.n_fg_class == 20

        if valid:
            YOLOv2Tiny(pretrained_model=self.pretrained_model, **params)
        else:
            with self.assertRaises(ValueError):
                YOLOv2Tiny(pretrained_model=self.pretrained_model, **params)


testing.run_module(__name__, __file__)
