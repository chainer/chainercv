import numpy as np
import unittest

import chainer
from chainer.cuda import to_cpu
from chainer.function import Function
from chainer import testing
from chainer.testing import attr

from chainercv.links import SequentialFeatureExtractor
from chainercv.utils.testing import ConstantStubLink


class DummyFunc(Function):

    def forward(self, inputs):
        return inputs[0] * 2,


@testing.parameterize(
    {'layer_names': None},
    {'layer_names': 'f2'},
    {'layer_names': ('f2',)},
    {'layer_names': ('l2', 'l1', 'f2')},
)
class TestSequentialFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.l1 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))
        self.f1 = DummyFunc()
        self.f2 = DummyFunc()
        self.l2 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))

        self.link = SequentialFeatureExtractor()
        with self.link.init_scope():
            self.link.l1 = self.l1
            self.link.f1 = self.f1
            self.link.f2 = self.f2
            self.link.l2 = self.l2

        if self.layer_names:
            self.link.layer_names = self.layer_names

        self.x = np.random.uniform(size=(1, 3, 24, 24))

    def check_call(self):
        x = self.link.xp.asarray(self.x)

        expects = dict()
        expects['l1'] = self.l1(x)
        expects['f1'] = self.f1(expects['l1'])
        expects['f2'] = self.f2(expects['f1'])
        expects['l2'] = self.l2(expects['f2'])

        outs = self.link(x)

        if isinstance(self.layer_names, tuple):
            layer_names = self.layer_names
        else:
            if self.layer_names is None:
                layer_names = ('l2',)
            else:
                layer_names = (self.layer_names,)
            outs = (outs,)

        self.assertEqual(len(outs), len(layer_names))

        for out, layer_name in zip(outs, layer_names):
            self.assertIsInstance(out, chainer.Variable)

            out = to_cpu(out.data)
            self.assertIsInstance(out, self.link.xp.ndarray)
            np.testing.assert_equal(out, to_cpu(expects[layer_name].data))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
