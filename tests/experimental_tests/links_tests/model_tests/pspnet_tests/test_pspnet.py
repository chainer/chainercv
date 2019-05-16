import numpy as np
import unittest

import chainer
from chainer import testing

from chainercv.experimental.links import PSPNetResNet101
from chainercv.experimental.links import PSPNetResNet50
from chainercv.utils import assert_is_semantic_segmentation_link
from chainercv.utils.testing import attr


@testing.parameterize(
    {'model': PSPNetResNet101},
    {'model': PSPNetResNet50},
)
class TestPSPNetResNet(unittest.TestCase):

    def setUp(self):
        self.n_class = 10
        self.input_size = (120, 160)
        params = self.model.preset_params['cityscapes'].copy()
        params['n_class'] = self.n_class
        params['input_size'] = self.input_size
        self.link = self.model(**params)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.random.uniform(
            low=-1, high=1, size=(2, 3) + self.input_size).astype(np.float32))
        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, xp.ndarray)
        self.assertEqual(y.shape, (2, self.n_class, 120, 160))

    @attr.slow
    @attr.pfnci_skip
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    @attr.slow
    @attr.pfnci_skip
    def test_predict_cpu(self):
        assert_is_semantic_segmentation_link(self.link, self.n_class)

    @attr.gpu
    @attr.slow
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_semantic_segmentation_link(self.link, self.n_class)


@testing.parameterize(*testing.product({
    'model': [PSPNetResNet50, PSPNetResNet101],
    'pretrained_model': ['cityscapes', 'ade20k', 'imagenet'],
    'n_class': [19, 150],
}))
class TestPSPNetResNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        params = self.model.preset_params['cityscapes'].copy()
        params['n_class'] = self.n_class

        if self.pretrained_model == 'cityscapes':
            valid = self.n_class == 19
        elif self.pretrained_model == 'ade20k':
            valid = self.n_class == 150
        elif self.pretrained_model == 'imagenet':
            valid = True

        if valid:
            self.model(pretrained_model=self.pretrained_model, **params)
        else:
            with self.assertRaises(ValueError):
                self.model(pretrained_model=self.pretrained_model, **params)


testing.run_module(__name__, __file__)
