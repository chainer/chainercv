import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr
from chainer import Variable

from chainercv.links import SEResNet101
from chainercv.links import SEResNet152
from chainercv.links import SEResNet50


@testing.parameterize(*(
    testing.product_dict(
        [
            {'pick': 'prob', 'shapes': (1, 200), 'n_class': 200},
            {'pick': 'res5',
             'shapes': (1, 2048, 7, 7), 'n_class': None},
            {'pick': ['res2', 'conv1'],
             'shapes': ((1, 256, 56, 56), (1, 64, 112, 112)), 'n_class': None},
        ],
        [
            {'model_class': SEResNet50},
            {'model_class': SEResNet101},
            {'model_class': SEResNet152},
        ],
    )
))
class TestSEResNetCall(unittest.TestCase):

    def setUp(self):
        self.link = self.model_class(
            n_class=self.n_class, pretrained_model=None)
        self.link.pick = self.pick

    def check_call(self):
        xp = self.link.xp

        x = Variable(xp.asarray(np.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(np.float32)))
        features = self.link(x)
        if isinstance(features, tuple):
            for activation, shape in zip(features, self.shapes):
                self.assertEqual(activation.shape, shape)
        else:
            self.assertEqual(features.shape, self.shapes)
            self.assertEqual(features.dtype, np.float32)

    @attr.slow
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


# TODO(g-votte): Add TestSEResNetPretrained.


testing.run_module(__name__, __file__)
