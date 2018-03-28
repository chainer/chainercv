import numpy as np
import unittest

from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


class TestProgressHook(unittest.TestCase):

    def setUp(self):
        def func(*in_values):
            n_sample = len(in_values[0])
            return [np.random.uniform() for _ in range(n_sample)]

        self.func = func

        self.dataset = []
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            self.dataset.append(np.random.randint(0, 256, size=(3, H, W)))

    def test_progress_hook(self):
        iterator = SerialIterator(self.dataset, 2, repeat=False)

        in_values, out_values, rest_values = apply_to_iterator(
            self.func, iterator,
            hook=ProgressHook(n_total=len(self.dataset)))

        # consume all data
        for _ in in_values[0]:
            pass

    def test_progress_hook_with_infinite_iterator(self):
        iterator = SerialIterator(self.dataset, 2)

        in_values, out_values, rest_values = apply_to_iterator(
            self.func, iterator, hook=ProgressHook())

        for _ in range(10):
            next(in_values[0])


testing.run_module(__name__, __file__)
