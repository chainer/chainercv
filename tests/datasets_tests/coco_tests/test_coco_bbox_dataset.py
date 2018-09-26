import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.utils import assert_is_bbox_dataset


def _create_paramters():
    split_years = testing.product({
        'split': ['train', 'val'],
        'year': ['2014', '2017']})
    split_years += [{'split': 'minival', 'year': '2014'},
                    {'split': 'valminusminival', 'year': '2014'}]
    use_and_return_args = testing.product({
        'use_crowded': [False, True],
        'return_crowded': [False, True],
        'return_area': [False, True]})
    params = testing.product_dict(
        split_years,
        use_and_return_args)
    return params


@testing.parameterize(*_create_paramters())
class TestCOCOBboxDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = COCOBboxDataset(
            split=self.split, year=self.year,
            use_crowded=self.use_crowded, return_area=self.return_area,
            return_crowded=self.return_crowded)

    @attr.slow
    def test_coco_bbox_dataset(self):
        assert_is_bbox_dataset(
            self.dataset, len(coco_bbox_label_names), n_example=30)

        if self.return_area:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                _, bbox, _, area = self.dataset[i][:4]
                self.assertIsInstance(area, np.ndarray)
                self.assertEqual(area.dtype, np.float32)
                self.assertEqual(area.shape, (bbox.shape[0],))

        if self.return_crowded:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                example = self.dataset[i]
                crowded = example[-1]
                bbox = example[1]
                self.assertIsInstance(crowded, np.ndarray)
                self.assertEqual(crowded.dtype, np.bool)
                self.assertEqual(crowded.shape, (bbox.shape[0],))

                if not self.use_crowded:
                    np.testing.assert_equal(crowded, 0)


testing.run_module(__name__, __file__)
