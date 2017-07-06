import collections
import unittest

import numpy as np

import chainer
from chainer.cuda import to_cpu
from chainer import testing
from chainer.testing import attr

from chainer.function import Function

from chainercv.links import SequentialFeatureExtractor
from chainercv.utils.testing import ConstantStubLink


class DummyFunc(Function):

    def forward(self, inputs):
        return inputs[0] * 2,


class TestSequentialFeatureExtractorOrderedDictFunctions(unittest.TestCase):

    def setUp(self):
        self.l1 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))
        self.f1 = DummyFunc()
        self.f2 = DummyFunc()
        self.l2 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))

        self.link = SequentialFeatureExtractor(
            collections.OrderedDict(
                [('l1', self.l1),
                 ('f1', self.f1),
                 ('f2', self.f2),
                 ('l2', self.l2)]),
            layer_names=['l1', 'f1', 'f2'])
        self.x = np.random.uniform(size=(1, 3, 24, 24))

    def check_call_output(self):
        x = self.link.xp.asarray(self.x)
        out = self.link(x)

        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[0], chainer.Variable)
        self.assertIsInstance(out[1], chainer.Variable)
        self.assertIsInstance(out[2], chainer.Variable)
        self.assertIsInstance(out[0].data, self.link.xp.ndarray)
        self.assertIsInstance(out[1].data, self.link.xp.ndarray)
        self.assertIsInstance(out[2].data, self.link.xp.ndarray)

        out_data = [to_cpu(var.data) for var in out]
        np.testing.assert_equal(out_data[0], to_cpu(self.l1(x).data))
        np.testing.assert_equal(out_data[1], to_cpu(self.f1(self.l1(x)).data))
        np.testing.assert_equal(
            out_data[2], to_cpu(self.f2(self.f1(self.l1(x))).data))

    def test_call_output_cpu(self):
        self.check_call_output()

    @attr.gpu
    def test_call_output_gpu(self):
        self.link.to_gpu()
        self.check_call_output()

    def check_call_dynamic_layer_names(self):
        x = self.link.xp.asarray(self.x)
        self.link.layer_names = ['l2']
        out, = self.link(x)

        self.assertIsInstance(out, chainer.Variable)
        self.assertIsInstance(out.data, self.link.xp.ndarray)

        out_data = out.data
        np.testing.assert_equal(
            out_data, to_cpu(self.l2(self.f2(self.f1(self.l1(x)))).data))

    def test_call_dynamic_layer_names_cpu(self):
        self.check_call_dynamic_layer_names()

    @attr.gpu
    def test_call_dynamic_layer_names_gpu(self):
        self.check_call_dynamic_layer_names()


class TestSequentialFeatureExtractorListFunctions(unittest.TestCase):

    def setUp(self):
        self.l1 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))
        self.f1 = DummyFunc()
        self.f2 = DummyFunc()
        self.l2 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))

        self.link = SequentialFeatureExtractor(
            [self.l1, self.f1, self.f2, self.l2])
        self.x = np.random.uniform(size=(1, 3, 24, 24))

    def check_call_output(self):
        x = self.link.xp.asarray(self.x)
        out = self.link(x)

        self.assertIsInstance(out, chainer.Variable)
        self.assertIsInstance(out.data, self.link.xp.ndarray)

        out = to_cpu(out.data)
        np.testing.assert_equal(
            out,
            to_cpu(self.l2(self.f2(self.f1(self.l1(x)))).data))

    def test_call_output_cpu(self):
        self.check_call_output()

    @attr.gpu
    def test_call_output_gpu(self):
        self.link.to_gpu()
        self.check_call_output()


class TestSequentialFeatureExtractorCopy(unittest.TestCase):

    def setUp(self):
        self.l1 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))
        self.f1 = DummyFunc()
        self.f2 = DummyFunc()
        self.l2 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))

        self.link = SequentialFeatureExtractor(
            collections.OrderedDict(
                [('l1', self.l1),
                 ('f1', self.f1),
                 ('f2', self.f2),
                 ('l2', self.l2)]),
            layer_names=['l1', 'f1', 'f2', 'l2'])

    def check_copy(self):
        copied = self.link.copy()
        self.assertIs(copied.l1, copied.layers['l1'])
        self.assertIs(copied.l2, copied.layers['l2'])

    def test_copy_cpu(self):
        self.check_copy()

    @attr.gpu
    def test_copy_gpu(self):
        self.link.to_gpu()
        self.check_copy()


testing.run_module(__name__, __file__)
