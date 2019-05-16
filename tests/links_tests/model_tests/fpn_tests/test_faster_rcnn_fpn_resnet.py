import numpy as np
import unittest

import chainer
from chainer import testing

from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv.links import MaskRCNNFPNResNet101
from chainercv.links import MaskRCNNFPNResNet50
from chainercv.utils.testing import attr


@testing.parameterize(*testing.product({
    'model': [FasterRCNNFPNResNet50, FasterRCNNFPNResNet101,
              MaskRCNNFPNResNet50, MaskRCNNFPNResNet101],
    'n_fg_class': [1, 5, 20],
}))
class TestFasterRCNNFPNResNet(unittest.TestCase):

    def setUp(self):
        params = self.model.preset_params['coco'].copy()
        params['n_fg_class'] = self.n_fg_class
        params['min_size'] = 66
        self.link = self.model(**params)

    def _check_call(self):
        imgs = [
            np.random.uniform(-1, 1, size=(3, 48, 48)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, 32, 64)).astype(np.float32),
        ]
        x, _ = self.link.prepare(imgs)
        with chainer.using_config('train', False):
            self.link(self.link.xp.array(x))

    @attr.slow
    @attr.pfnci_skip
    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()


@testing.parameterize(*testing.product({
    'model': [FasterRCNNFPNResNet50, FasterRCNNFPNResNet101,
              MaskRCNNFPNResNet50, MaskRCNNFPNResNet101],
    'n_fg_class': [10, 80],
    'pretrained_model': ['coco', 'imagenet'],
}))
class TestFasterRCNNFPNResNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        params = self.model.preset_params['coco'].copy()
        params['n_fg_class'] = self.n_fg_class

        if self.pretrained_model == 'coco':
            valid = self.n_fg_class == 80
        elif self.pretrained_model == 'imagenet':
            valid = True

        if valid:
            self.model(pretrained_model=self.pretrained_model, **params)
        else:
            with self.assertRaises(ValueError):
                self.model(pretrained_model=self.pretrained_model, **params)


testing.run_module(__name__, __file__)
