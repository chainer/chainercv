import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import coco_keypoint_names
from chainercv.datasets import COCOKeypointDataset
from chainercv.utils import assert_is_bbox
from chainercv.utils import assert_is_point_dataset


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


@testing.parameterize(*testing.product(
    {
        'split': ['train', 'val'],
        'year': ['2014', '2017'],
        'use_crowded': [False, True],
        'return_crowded': [False, True],
        'return_area': [False, True],
    }
))
class TestCOCOKeypointDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = COCOKeypointDataset(
            split=self.split, year=self.year,
            use_crowded=self.use_crowded, return_area=self.return_area,
            return_crowded=self.return_crowded)

    @attr.slow
    def test_coco_keypoint_dataset(self):
        human_id = 0
        assert_is_point_dataset(
            self.dataset, len(coco_keypoint_names[human_id]),
            n_example=30)

        for _ in range(10):
            i = np.random.randint(0, len(self.dataset))
            img, point, _, label, bbox = self.dataset[i][:5]
            assert_is_bbox(bbox, img.shape[1:])
            self.assertEqual(len(bbox), len(point))

            self.assertIsInstance(label, np.ndarray)
            self.assertEqual(label.dtype, np.int32)
            self.assertEqual(label.shape, (point.shape[0],))

        if self.return_area:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                _, point, _, _, _, area = self.dataset[i][:6]
                self.assertIsInstance(area, np.ndarray)
                self.assertEqual(area.dtype, np.float32)
                self.assertEqual(area.shape, (point.shape[0],))

        if self.return_crowded:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                example = self.dataset[i]
                crowded = example[-1]
                point = example[1]
                self.assertIsInstance(crowded, np.ndarray)
                self.assertEqual(crowded.dtype, np.bool)
                self.assertEqual(crowded.shape, (point.shape[0],))

                if not self.use_crowded:
                    np.testing.assert_equal(crowded, 0)


testing.run_module(__name__, __file__)
