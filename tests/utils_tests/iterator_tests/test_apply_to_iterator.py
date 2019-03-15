import numpy as np
from six.moves import zip_longest
import unittest

import chainer
from chainer.iterators import SerialIterator
from chainer import testing
from chainermn import create_communicator

from chainercv.utils import apply_to_iterator
from chainercv.utils.testing import attr


@testing.parameterize(*testing.product({
    'multi_in_values': [False, True],
    'multi_out_values': [False, True],
    'with_rest_values': [False, True],
    'with_hook': [False, True],
}))
class TestApplyToIterator(unittest.TestCase):

    def setUp(self):
        if self.multi_in_values:
            self.n_input = 2
        else:
            self.n_input = 1

        in_values_expect = []
        for _ in range(self.n_input):
            in_value = []
            for _ in range(5):
                H, W = np.random.randint(8, 16, size=2)
                in_value.append(np.random.randint(0, 256, size=(3, H, W)))
            in_values_expect.append(in_value)
        self.in_values_expect = tuple(in_values_expect)

        if self.multi_out_values:
            def func(*in_values):
                n_sample = len(in_values[0])
                return (
                    [np.random.uniform(size=(10, 4)) for _ in range(n_sample)],
                    [np.random.uniform(size=10) for _ in range(n_sample)],
                    [np.random.uniform(size=10) for _ in range(n_sample)])

            self.n_output = 3
        else:
            def func(*in_values):
                n_sample = len(in_values[0])
                return [np.random.uniform(size=(48, 64))
                        for _ in range(n_sample)]

            self.n_output = 1
        self.func = func

        if self.with_rest_values:
            strs = ['a', 'bc', 'def', 'ghij', 'klmno']
            nums = [0, 1, 2, 3, 4]
            arrays = [np.random.uniform(size=10) for _ in range(5)]
            self.rest_values_expect = (strs, nums, arrays)
            self.n_rest = 3

            self.dataset = chainer.datasets.TupleDataset(
                *(self.in_values_expect + self.rest_values_expect))
        else:
            self.rest_values_expect = ()
            self.n_rest = 0

            self. dataset = list(zip(*self.in_values_expect))

        self.iterator = SerialIterator(
            self.dataset, 2, repeat=False, shuffle=False)

        if self.with_hook:
            def hook(in_values, out_values, rest_values):
                n_sample = len(in_values[0])

                self.assertEqual(len(in_values), self.n_input)
                for in_vals in in_values:
                    self.assertEqual(len(in_vals), n_sample)

                self.assertEqual(len(out_values), self.n_output)
                for out_vals in out_values:
                    self.assertEqual(len(out_vals), n_sample)

                self.assertEqual(len(rest_values), self.n_rest)
                for rest_vals in rest_values:
                    self.assertEqual(len(rest_vals), n_sample)

            self.hook = hook
        else:
            self.hook = None

    def _check_apply_to_iterator(self, comm=None):
        values = apply_to_iterator(
            self.func, self.iterator, n_input=self.n_input,
            hook=self.hook, comm=comm)

        if comm is not None and not comm.rank == 0:
            self.assertEqual(values, None)
            return

        in_values, out_values, rest_values = values

        self.assertEqual(len(in_values), self.n_input)
        for in_vals, in_vals_expect in \
                zip_longest(in_values, self.in_values_expect):
            for in_val, in_val_expect in zip_longest(in_vals, in_vals_expect):
                np.testing.assert_equal(in_val, in_val_expect)

        self.assertEqual(len(out_values), self.n_output)
        for out_vals in out_values:
            self.assertEqual(len(list(out_vals)), len(self.dataset))

        self.assertEqual(len(rest_values), self.n_rest)
        for rest_vals, rest_vals_expect in \
                zip_longest(rest_values, self.rest_values_expect):
            for rest_val, rest_val_expect in \
                    zip_longest(rest_vals, rest_vals_expect):
                if isinstance(rest_val_expect, np.ndarray):
                    np.testing.assert_equal(rest_val, rest_val_expect)
                else:
                    self.assertEqual(rest_val, rest_val_expect)

    def test_apply_to_iterator(self):
        self._check_apply_to_iterator()

    @attr.mpi
    def test_apply_to_iterator_with_comm(self):
        comm = create_communicator('naive')
        self._check_apply_to_iterator(comm)


class TestApplyToIteratorWithInfiniteIterator(unittest.TestCase):

    def test_apply_to_iterator_with_infinite_iterator(self):
        def func(*in_values):
            n_sample = len(in_values[0])
            return [np.random.uniform(size=(48, 64)) for _ in range(n_sample)]

        dataset = []
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            dataset.append(np.random.randint(0, 256, size=(3, H, W)))

        iterator = SerialIterator(dataset, 2)

        in_values, out_values, rest_values = apply_to_iterator(func, iterator)

        for _ in range(10):
            next(in_values[0])

        for _ in range(10):
            next(out_values[0])


testing.run_module(__name__, __file__)
