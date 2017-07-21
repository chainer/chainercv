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
    {'feature_names': None},
    {'feature_names': 'f2'},
    {'feature_names': ('f2',)},
    {'feature_names': ('l2', 'l1', 'f2')},
    {'feature_names': ('l2', 'l2')},
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

        if self.feature_names:
            self.link.feature_names = self.feature_names

        self.x = np.random.uniform(size=(1, 3, 24, 24))

    def test_feature_names(self):
        self.assertEqual(self.link.feature_names, self.feature_names)

    def test_all_feature_names(self):
        self.assertEqual(self.link.all_feature_names, ['l1', 'f1', 'f2', 'l2'])

    def test_index(self):
        self.assertEqual(self.link.index('l1'), 0)
        self.assertEqual(self.link.index('f1'), 1)
        self.assertEqual(self.link.index('f2'), 2)
        self.assertEqual(self.link.index('l2'), 3)

    def check_call(self, x, expects):
        outs = self.link(x)

        if isinstance(self.feature_names, tuple):
            feature_names = self.feature_names
        else:
            if self.feature_names is None:
                feature_names = ('l2',)
            else:
                feature_names = (self.feature_names,)
            outs = (outs,)

        self.assertEqual(len(outs), len(feature_names))

        for out, layer_name in zip(outs, feature_names):
            self.assertIsInstance(out, chainer.Variable)
            self.assertIsInstance(out.data, self.link.xp.ndarray)

            out = to_cpu(out.data)
            np.testing.assert_equal(out, to_cpu(expects[layer_name].data))

    def check_basic(self):
        x = self.link.xp.asarray(self.x)

        expects = dict()
        expects['l1'] = self.l1(x)
        expects['f1'] = self.f1(expects['l1'])
        expects['f2'] = self.f2(expects['f1'])
        expects['l2'] = self.l2(expects['f2'])

        self.check_call(x, expects)

    def test_basic_cpu(self):
        self.check_basic()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_basic()

    def check_deletion(self):
        x = self.link.xp.asarray(self.x)

        if self.feature_names == 'l1' or \
           (isinstance(self.feature_names, tuple) and
                'l1' in self.feature_names):
            with self.assertRaises(AttributeError):
                del self.link.l1
            return
        else:
            del self.link.l1

        expects = dict()
        expects['f1'] = self.f1(x)
        expects['f2'] = self.f2(expects['f1'])
        expects['l2'] = self.l2(expects['f2'])

        self.check_call(x, expects)

    def test_deletion_cpu(self):
        self.check_deletion()

    @attr.gpu
    def test_deletion_gpu(self):
        self.link.to_gpu()
        self.check_deletion()


@testing.parameterize(
    {'feature_names': 'f1', 'index': 1, 'all_feature_names': None},
    {'feature_names': 'f1', 'index': slice(None, 2),
     'all_feature_names': ['l1', 'f1']},
    {'feature_names': 'f2', 'index': slice(2, -1), 'all_feature_names': ['f2']},
    {'feature_names': 'l1', 'index': slice(None, 2, 2), 'all_feature_names': ['l1']}
)
class TestSequentialFeatureExtractorGetitem(unittest.TestCase):

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
        self.link.feature_names = self.feature_names

    def check_getitem(self):
        ret = self.link[self.index]
        if isinstance(self.index, int):
            expected_type = type(getattr(
                self.link, self.link.all_feature_names[self.index]))
            self.assertIsInstance(ret, expected_type)
        elif isinstance(self.index, slice):
            self.assertEqual(ret.all_feature_names, self.all_feature_names)

    def test_getitem_cpu(self):
        self.check_getitem()

    @attr.gpu
    def test_getitem_gpu(self):
        self.link.to_gpu()
        self.check_getitem()


testing.run_module(__name__, __file__)
