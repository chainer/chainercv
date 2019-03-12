import unittest

from chainer import testing

from chainercv.chainer_experimental.datasets.sliceable import SliceableDataset
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset


class SampleDataset(SliceableDataset):

    def __len__(self):
        return 10

    @property
    def keys(self):
        return ('item0', 'item1', 'item2')

    def get_example_by_keys(self, i, key_indices):
        return tuple(
            '{:s}({:d})'.format(self.keys[key_index], i)
            for key_index in key_indices)


@testing.parameterize(*testing.product_dict(
    [
        {'iterable': tuple},
        {'iterable': list},
    ],
    [
        {
            'keys': 'item1',
            'func': lambda in_data: 'transformed_' + in_data[1],
            'expected_sample': 'transformed_item1(3)',
        },
        {
            'keys': ('item1',),
            'func': lambda in_data: ('transformed_' + in_data[1],),
            'expected_sample': ('transformed_item1(3)',),
        },
        {
            'keys': ('item0', 'item2'),
            'func': lambda in_data: (
                'transformed_' + in_data[0],
                'transformed_' + in_data[2]),
            'expected_sample':
            ('transformed_item0(3)', 'transformed_item2(3)'),
        },
    ],
))
class TestTransformDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = SampleDataset()

    def _check(self, dataset, expected_keys):
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), len(self.dataset))
        self.assertEqual(dataset.keys, expected_keys)
        self.assertEqual(dataset[3], self.expected_sample)

    def test_transform(self):
        if isinstance(self.keys, tuple):
            keys = self.iterable(self.keys)
        else:
            keys = self.keys
        dataset = TransformDataset(self.dataset, keys, self.func)
        self._check(dataset, self.keys)

    def test_transform_with_n_keys(self):
        if isinstance(self.keys, tuple):
            n_keys = len(self.keys)
            if n_keys == 1:
                self.skipTest(
                    'tuple of single element is invalid '
                    'when the number of keys is specified')
            expected_keys = (None,) * n_keys
        else:
            n_keys = 1
            expected_keys = None
        dataset = TransformDataset(self.dataset, n_keys, self.func)
        self._check(dataset, expected_keys)

    def test_transform_compat(self):
        if isinstance(self.keys, tuple):
            expected_keys = (None,) * len(self.keys)
        else:
            expected_keys = None
        dataset = TransformDataset(self.dataset, self.func)
        self._check(dataset, expected_keys)


testing.run_module(__name__, __file__)
