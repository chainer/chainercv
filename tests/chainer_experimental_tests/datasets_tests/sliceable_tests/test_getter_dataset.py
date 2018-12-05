import numpy as np
import unittest

from chainer import testing

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset


class SampleDataset(GetterDataset):
    def __init__(self, iterable=tuple):
        super(SampleDataset, self).__init__()

        self.add_getter('item0', self.get_item0)
        self.add_getter(iterable(('item1', 'item2')), self.get_item1_item2)
        self.add_getter(1, self.get_item3)

        self.count = 0

    def __len__(self):
        return 10

    def get_item0(self, i):
        self.count += 1
        return 'item0({:d})'.format(i)

    def get_item1_item2(self, i):
        self.count += 1
        return 'item1({:d})'.format(i), 'item2({:d})'.format(i)

    def get_item3(self, i):
        self.count += 1
        return 'item3({:d})'.format(i)


@testing.parameterize(
    {'iterable': tuple},
    {'iterable': list},
    {'iterable': np.array},
)
class TestGetterDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = SampleDataset(self.iterable)

    def test_keys(self):
        self.assertEqual(
            self.dataset.keys, ('item0', 'item1', 'item2', None))

    def test_get_example_by_keys(self):
        example = self.dataset.get_example_by_keys(1, (1, 2, 3))
        self.assertEqual(example, ('item1(1)', 'item2(1)', 'item3(1)'))
        self.assertEqual(self.dataset.count, 2)

    def test_set_keys_single_name(self):
        self.dataset.keys = 'item0'
        self.assertEqual(self.dataset.keys, 'item0')
        self.assertEqual(self.dataset[1], 'item0(1)')

    def test_set_keys_single_index(self):
        self.dataset.keys = 0
        self.assertEqual(self.dataset.keys, 'item0')
        self.assertEqual(self.dataset[1], 'item0(1)')

    def test_set_keys_single_tuple_name(self):
        if self.iterable is np.array:
            self.skipTest('ndarray of strings is not supported')
        self.dataset.keys = self.iterable(('item1',))
        self.assertEqual(self.dataset.keys, ('item1',))
        self.assertEqual(self.dataset[2], ('item1(2)',))

    def test_set_keys_single_tuple_index(self):
        self.dataset.keys = self.iterable((1,))
        self.assertEqual(self.dataset.keys, ('item1',))
        self.assertEqual(self.dataset[2], ('item1(2)',))

    def test_set_keys_multiple_name(self):
        if self.iterable is np.array:
            self.skipTest('ndarray of strings is not supported')
        self.dataset.keys = self.iterable(('item0', 'item2'))
        self.assertEqual(self.dataset.keys, ('item0', 'item2'))
        self.assertEqual(self.dataset[3], ('item0(3)', 'item2(3)'))

    def test_set_keys_multiple_index(self):
        self.dataset.keys = self.iterable((0, 2))
        self.assertEqual(self.dataset.keys, ('item0', 'item2'))
        self.assertEqual(self.dataset[3], ('item0(3)', 'item2(3)'))

    def test_set_keys_multiple_bool(self):
        self.dataset.keys = self.iterable((True, False, True, False))
        self.assertEqual(self.dataset.keys, ('item0', 'item2'))
        self.assertEqual(self.dataset[3], ('item0(3)', 'item2(3)'))

    def test_set_keys_multiple_mixed(self):
        if self.iterable is np.array:
            self.skipTest('ndarray of strings is not supported')
        self.dataset.keys = self.iterable(('item0', 2))
        self.assertEqual(self.dataset.keys, ('item0', 'item2'))
        self.assertEqual(self.dataset[3], ('item0(3)', 'item2(3)'))

    def test_set_keys_invalid_name(self):
        with self.assertRaises(KeyError):
            self.dataset.keys = 'invalid'

    def test_set_keys_invalid_index(self):
        with self.assertRaises(IndexError):
            self.dataset.keys = 4

    def test_set_keys_invalid_bool(self):
        with self.assertRaises(ValueError):
            self.dataset.keys = (True, True)


testing.run_module(__name__, __file__)
