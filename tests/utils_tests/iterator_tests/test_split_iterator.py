import random
import unittest

from chainer import testing

from chainercv.utils.iterator import split_iterator


class TestSplitIterator(unittest.TestCase):

    def setUp(self):
        self.ints = list(range(10))
        self.strs = list('abcdefghij')
        self.iterator = iter(zip(self.ints, self.strs))

    def test_sequential(self):
        i_iter, s_iter = split_iterator(self.iterator)

        ints = list(i_iter)
        self.assertEqual(ints, self.ints)

        strs = list(s_iter)
        self.assertEqual(strs, self.strs)

    def test_parallel(self):
        i_iter, s_iter = split_iterator(self.iterator)

        ints, strs = list(), list()
        for i, s in zip(i_iter, s_iter):
            ints.append(i)
            strs.append(s)

        self.assertEqual(ints, self.ints)
        self.assertEqual(strs, self.strs)

    def test_random(self):
        i_iter, s_iter = split_iterator(self.iterator)

        ints, strs = list(), list()
        while True:
            try:
                if random.randrange(2):
                    ints.append(next(i_iter))
                else:
                    strs.append(next(s_iter))
            except StopIteration:
                break

        ints.extend(i_iter)
        strs.extend(s_iter)

        self.assertEqual(ints, self.ints)
        self.assertEqual(strs, self.strs)


class TestSplitIteratorWithInfiniteIterator(unittest.TestCase):

    def setUp(self):

        def _iterator():
            i = 0
            while True:
                yield i, i + 1, i * i
                i += 1

        self.iterator = _iterator()

    def test_sequential(self):
        iters = split_iterator(self.iterator)

        self.assertEqual(len(iters), 3)

        for i in range(10):
            self.assertEqual(next(iters[0]), i)

        for i in range(10):
            self.assertEqual(next(iters[1]), i + 1)

        for i in range(10):
            self.assertEqual(next(iters[2]), i * i)

    def test_parallel(self):
        iters = split_iterator(self.iterator)

        self.assertEqual(len(iters), 3)

        for i in range(10):
            self.assertEqual(next(iters[0]), i)
            self.assertEqual(next(iters[1]), i + 1)
            self.assertEqual(next(iters[2]), i * i)


testing.run_module(__name__, __file__)
