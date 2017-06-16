import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset
from chainercv.utils import assert_is_detection_dataset


def _create_paramters():
    split_years = testing.product({
        'split': ['train', 'trainval', 'val'],
        'year': ['2007', '2012']})
    split_years += [{'split': 'test', 'year': '2007'}]
    params = testing.product_dict(
        split_years,
        [{'use_difficult': True, 'return_difficult': True},
         {'use_difficult': True, 'return_difficult': False},
         {'use_difficult': False, 'return_difficult': True},
         {'use_difficult': False, 'return_difficult': False}])
    return params


@testing.parameterize(*_create_paramters())
class TestVOCDetectionDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = VOCDetectionDataset(
            split=self.split,
            year=self.year,
            use_difficult=self.use_difficult,
            return_difficult=self.return_difficult)
        self.n_out = 4 if self.return_difficult else 3

    @attr.slow
    def test_as_detection_dataset(self):
        assert_is_detection_dataset(
            self.dataset, len(voc_detection_label_names), n_example=10)

    @attr.slow
    @condition.repeat(10)
    def test_difficult(self):
        if not self.return_difficult:
            return

        i = np.random.randint(0, len(self.dataset))
        _, bbox, _, difficult = self.dataset[i]
        self.assertIsInstance(difficult, np.ndarray)
        self.assertEqual(difficult.dtype, np.bool)
        self.assertEqual(difficult.shape, (bbox.shape[0],))

        if not self.use_difficult:
            np.testing.assert_equal(difficult, 0)


testing.run_module(__name__, __file__)
