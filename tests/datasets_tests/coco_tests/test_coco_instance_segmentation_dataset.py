import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.datasets import COCOInstanceSegmentationDataset
from chainercv.utils import assert_is_bbox
from chainercv.utils import assert_is_instance_segmentation_dataset

try:
    import pycocotools  # NOQA
    _available = True
except ImportError:
    _available = False


def _create_paramters():
    split_years = testing.product({
        'split': ['train', 'val'],
        'year': ['2014', '2017']})
    split_years += [{'split': 'minival', 'year': '2014'},
                    {'split': 'valminusminival', 'year': '2014'}]
    use_and_return_args = testing.product({
        'use_crowded': [False, True],
        'return_crowded': [False, True],
        'return_area': [False, True],
        'return_bbox': [False, True]})
    params = testing.product_dict(
        split_years,
        use_and_return_args)
    return params


@testing.parameterize(*_create_paramters())
class TestCOCOInstanceSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = COCOInstanceSegmentationDataset(
            split=self.split, year=self.year,
            use_crowded=self.use_crowded, return_crowded=self.return_crowded,
            return_area=self.return_area, return_bbox=self.return_bbox)

    @attr.slow
    @unittest.skipUnless(_available, 'pycocotools is not installed')
    def test_coco_instance_segmentation_dataset(self):
        assert_is_instance_segmentation_dataset(
            self.dataset,
            len(coco_instance_segmentation_label_names),
            n_example=10)

        if self.return_area:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                _, mask, _, area = self.dataset[i][:4]
                self.assertIsInstance(area, np.ndarray)
                self.assertEqual(area.dtype, np.float32)
                self.assertEqual(area.shape, (mask.shape[0],))

        if self.return_crowded:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                example = self.dataset[i]
                if self.return_area:
                    crowded = example[4]
                else:
                    crowded = example[3]
                mask = example[1]
                self.assertIsInstance(crowded, np.ndarray)
                self.assertEqual(crowded.dtype, np.bool)
                self.assertEqual(crowded.shape, (mask.shape[0],))

                if not self.use_crowded:
                    np.testing.assert_equal(crowded, 0)

        if self.return_bbox:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                example = self.dataset[i]
                bbox = example[-1]
                img, mask = example[:2]
                assert_is_bbox(bbox, img.shape[1:])
                self.assertEqual(len(bbox), len(mask))


testing.run_module(__name__, __file__)
