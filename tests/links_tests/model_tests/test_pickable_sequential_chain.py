import numpy as np
import unittest

import chainer
from chainer.backends.cuda import to_cpu
from chainer.function import Function
from chainer import testing
from chainer.testing import attr

from chainercv.links import PickableSequentialChain
from chainercv.utils.testing import ConstantStubLink


class DummyFunc(Function):

    def forward(self, inputs):
        return inputs[0] * 2,


class PickableSequentialChainTestBase(object):

    def setUpBase(self):
        self.l1 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))
        self.f1 = DummyFunc()
        self.f2 = DummyFunc()
        self.l2 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))

        self.link = PickableSequentialChain()
        with self.link.init_scope():
            self.link.l1 = self.l1
            self.link.f1 = self.f1
            self.link.f2 = self.f2
            self.link.l2 = self.l2

        if self.pick:
            self.link.pick = self.pick

        self.x = np.random.uniform(size=(1, 3, 24, 24))

    def test_pick(self):
        self.assertEqual(self.link.pick, self.pick)

    def test_layer_names(self):
        self.assertEqual(self.link.layer_names, ['l1', 'f1', 'f2', 'l2'])

    def check_call(self, x, expects):
        outs = self.link(x)

        if isinstance(self.pick, tuple):
            pick = self.pick
        else:
            if self.pick is None:
                pick = ('l2',)
            else:
                pick = (self.pick,)
            outs = (outs,)

        self.assertEqual(len(outs), len(pick))

        for out, layer_name in zip(outs, pick):
            self.assertIsInstance(out, chainer.Variable)
            self.assertIsInstance(out.array, self.link.xp.ndarray)
            self.assertEqual(out.name, layer_name)

            out = to_cpu(out.array)
            np.testing.assert_equal(out, to_cpu(expects[layer_name].array))

    def check_basic(self):
        x = self.link.xp.asarray(self.x)

        expects = {}
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

        if self.pick == 'l1' or \
           (isinstance(self.pick, tuple)
                and 'l1' in self.pick):
            with self.assertRaises(AttributeError):
                del self.link.l1
            return
        else:
            del self.link.l1

        expects = {}
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
    {'pick': None},
    {'pick': 'f2'},
    {'pick': ('f2',)},
    {'pick': ('l2', 'l1', 'f2')},
    {'pick': ('l2', 'l2')},
)
class TestPickableSequentialChain(
        unittest.TestCase, PickableSequentialChainTestBase):
    def setUp(self):
        self.setUpBase()


@testing.parameterize(
    *testing.product({
        'mode': ['init', 'share', 'copy'],
        'pick': [None, 'f1', ('f1', 'f2'), ('l2', 'l2'), ('l2', 'l1', 'f2')]
    })
)
class TestCopiedPickableSequentialChain(
        unittest.TestCase, PickableSequentialChainTestBase):

    def setUp(self):
        self.setUpBase()

        self.f100 = DummyFunc()
        self.l100 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))

        self.link, self.original_link = \
            self.link.copy(mode=self.mode), self.link

    def check_unchanged(self, link, x):
        class Checker(object):
            def __init__(self, tester, link, x):
                self.tester = tester
                self.link = link
                self.x = x

            def __enter__(self):
                self.expected = self.link(self.x)

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_type is not None:
                    return None

                self.actual = self.link(self.x)

                if isinstance(self.expected, tuple):
                    self.tester.assertEqual(
                        len(self.expected), len(self.actual))
                    for e, a in zip(self.expected, self.actual):
                        self.tester.assertEqual(type(e.array), type(a.array))
                        np.testing.assert_equal(
                            to_cpu(e.array), to_cpu(a.array))
                else:
                    self.tester.assertEqual(type(self.expected.array),
                                            type(self.actual.array))
                    np.testing.assert_equal(
                        to_cpu(self.expected.array),
                        to_cpu(self.actual.array))

        return Checker(self, link, x)

    def test_original_unaffected_by_setting_pick(self):
        with self.check_unchanged(self.original_link, self.x):
            self.link.pick = 'f2'

    def test_original_unaffected_by_function_addition(self):
        with self.check_unchanged(self.original_link, self.x):
            with self.link.init_scope():
                self.link.f100 = self.f100

    def test_original_unaffected_by_link_addition(self):
        with self.check_unchanged(self.original_link, self.x):
            with self.link.init_scope():
                self.link.l100 = self.l100

    def test_original_unaffected_by_function_deletion(self):
        with self.check_unchanged(self.original_link, self.x):
            with self.link.init_scope():
                self.link.pick = None
                del self.link.f1

    def test_original_unaffected_by_link_deletion(self):
        with self.check_unchanged(self.original_link, self.x):
            with self.link.init_scope():
                self.link.pick = None
                del self.link.l1


@testing.parameterize(
    {'pick': 'l1', 'layer_names': ['l1']},
    {'pick': 'f1', 'layer_names': ['l1', 'f1']},
    {'pick': ['f1', 'f2'], 'layer_names': ['l1', 'f1', 'f2']},
    {'pick': None, 'layer_names': ['l1', 'f1', 'f2', 'l2']}
)
class TestPickableSequentialChainRemoveUnused(unittest.TestCase):

    def setUp(self):
        self.l1 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))
        self.f1 = DummyFunc()
        self.f2 = DummyFunc()
        self.l2 = ConstantStubLink(np.random.uniform(size=(1, 3, 24, 24)))

        self.link = PickableSequentialChain()
        with self.link.init_scope():
            self.link.l1 = self.l1
            self.link.f1 = self.f1
            self.link.f2 = self.f2
            self.link.l2 = self.l2
        self.link.pick = self.pick

    def check_remove_unused(self):
        self.link.remove_unused()

        self.assertEqual(self.link.layer_names, self.layer_names)
        for name in ['l1', 'f1', 'f2', 'l2']:
            if name in self.layer_names:
                self.assertTrue(hasattr(self.link, name))
            else:
                self.assertFalse(hasattr(self.link, name))

    def test_remove_unused_cpu(self):
        self.check_remove_unused()

    @attr.gpu
    def test_remove_unused_gpu(self):
        self.link.to_gpu()
        self.check_remove_unused()


testing.run_module(__name__, __file__)
