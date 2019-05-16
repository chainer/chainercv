import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import SSD300
from chainercv.links import SSD512


@testing.parameterize(*testing.product({
    'insize': [300, 512],
    'n_fg_class': [1, 5, 20],
}))
class TestSSDVGG16(unittest.TestCase):

    def setUp(self):
        if self.insize == 300:
            param = SSD300.preset_params['voc'].copy()
            param['n_fg_class'] = self.n_fg_class
            self.link = SSD300(**param)
            self.n_bbox = 8732
        elif self.insize == 512:
            param = SSD300.preset_params['voc'].copy()
            param['n_fg_class'] = self.n_fg_class
            self.link = SSD512(**param)
            self.n_bbox = 24564

    def _check_call(self):
        x = self.link.xp.array(
            np.random.uniform(-1, 1, size=(1, 3, self.insize, self.insize)),
            dtype=np.float32)

        loc, conf = self.link(x)

        self.assertIsInstance(loc, chainer.Variable)
        self.assertIsInstance(loc.array, self.link.xp.ndarray)
        self.assertEqual(loc.shape, (1, self.n_bbox, 4))
        self.assertIsInstance(conf, chainer.Variable)
        self.assertIsInstance(conf.array, self.link.xp.ndarray)
        self.assertEqual(conf.shape, (1, self.n_bbox, self.n_fg_class + 1))

    @attr.slow
    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()


@testing.parameterize(*testing.product({
    'model': [SSD300, SSD512],
    'n_fg_class': [10, 20],
    'pretrained_model': ['voc0712', 'imagenet'],
}))
class TestSSDVGG16Pretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        param = self.model.preset_params['voc'].copy()
        param['n_fg_class'] = self.n_fg_class

        if self.pretrained_model == 'voc0712':
            valid = self.n_fg_class == 20
        elif self.pretrained_model == 'imagenet':
            valid = True

        if valid:
            self.model(pretrained_model=self.pretrained_model, **param)
        else:
            with self.assertRaises(ValueError):
                self.model(pretrained_model=self.pretrained_model, **param)


testing.run_module(__name__, __file__)
