import random
import unittest

from chainer import testing

from chainercv.utils.iterator import split_iterator


class TestSplitIterator(unittest.TestCase):

    def setUp(self):
        self.ints = list(range(10))
        self.strs = list('abcdefghij')

        def _iterator():
            for i, s in zip(self.ints, self.strs):
                yield i, s

        self.iterator = _iterator()

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
            except:
                break

        ints.extend(i_iter)
        strs.extend(s_iter)

        self.assertEqual(ints, self.ints)
        self.assertEqual(strs, self.strs)


testing.run_module(__name__, __file__)
