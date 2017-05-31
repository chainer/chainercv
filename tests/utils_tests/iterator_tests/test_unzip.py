import random
import unittest

from chainer import testing

from chainercv.utils import unzip


class TestUnzip(unittest.TestCase):

    def setUp(self):
        self.ints = list(range(10))
        self.strs = list('abcdefghij')
        self.iterable = zip(self.ints, self.strs)

    def test_sequential(self):
        i_iter, s_iter = unzip(self.iterable)

        ints = list(i_iter)
        self.assertEqual(ints, self.ints)

        strs = list(s_iter)
        self.assertEqual(strs, self.strs)

    def test_parallel(self):
        i_iter, s_iter = unzip(self.iterable)

        ints, strs = list(), list()
        for i, s in zip(i_iter, s_iter):
            ints.append(i)
            strs.append(s)

        self.assertEqual(ints, self.ints)
        self.assertEqual(strs, self.strs)

    def test_random(self):
        i_iter, s_iter = unzip(self.iterable)

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


class TestUnzipWithInfiniteIterator(unittest.TestCase):

    def setUp(self):

        def _iterator():
            i = 0
            while True:
                yield i, i + 1, i * i
                i += 1

        self.iterable = _iterator()

    def test_sequential(self):
        iters = unzip(self.iterable)

        self.assertEqual(len(iters), 3)

        for i in range(10):
            self.assertEqual(next(iters[0]), i)

        for i in range(10):
            self.assertEqual(next(iters[1]), i + 1)

        for i in range(10):
            self.assertEqual(next(iters[2]), i * i)

    def test_parallel(self):
        iters = unzip(self.iterable)

        self.assertEqual(len(iters), 3)

        for i in range(10):
            self.assertEqual(next(iters[0]), i)
            self.assertEqual(next(iters[1]), i + 1)
            self.assertEqual(next(iters[2]), i * i)


class DummyObject(object):

    def __init__(self, released, id_):
        self.released = released
        self.id_ = id_

    def __del__(self):
        # register id when it is released
        self.released.add(self.id_)


class TestUnzipRelease(unittest.TestCase):

    def setUp(self):
        self.released = set()

        def _iterator():
            id_ = 0
            while True:
                yield id_, DummyObject(self.released, id_)
                id_ += 1

        self.iterable = _iterator()

    def test_released(self):
        iter_0, iter_1 = unzip(self.iterable)
        del iter_1

        for i in range(20):
            next(iter_0)

        self.assertEqual(self.released, set(range(20)))

    def test_unreleased(self):
        iter_0, iter_1 = unzip(self.iterable)

        for i in range(20):
            next(iter_0)

        self.assertEqual(self.released, set())


testing.run_module(__name__, __file__)
