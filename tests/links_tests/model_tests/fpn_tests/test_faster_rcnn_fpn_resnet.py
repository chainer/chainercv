import numpy as np
import unittest

import chainer
from chainer import testing

from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv.utils.testing import attr


@testing.parameterize(*testing.product({
    'model': [FasterRCNNFPNResNet50, FasterRCNNFPNResNet101],
    'n_fg_class': [1, 5, 20],
}))
class TestFasterRCNNFPNResNet(unittest.TestCase):

    def setUp(self):
        self.link = self.model(n_fg_class=self.n_fg_class)

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
    'model': [FasterRCNNFPNResNet50, FasterRCNNFPNResNet101],
    'n_fg_class': [None, 10, 80],
    'pretrained_model': ['coco', 'imagenet'],
}))
class TestFasterRCNNFPNResNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_fg_class': self.n_fg_class,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model == 'coco':
            valid = self.n_fg_class in {None, 80}
        elif self.pretrained_model == 'imagenet':
            valid = self.n_fg_class is not None

        if valid:
            self.model(**kwargs)
        else:
            with self.assertRaises(ValueError):
                self.model(**kwargs)


testing.run_module(__name__, __file__)
