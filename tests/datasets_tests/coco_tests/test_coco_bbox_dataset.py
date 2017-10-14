import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.utils import assert_is_bbox_dataset


@testing.parameterize(*testing.product({
    'split': ['train', 'val', 'minival', 'valminusminival'],
    'use_crowded': [False, True],
    'return_crowded': [False, True],
    'return_area': [False, True]
}))
class TestCOCOBboxDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = COCOBboxDataset(
            split=self.split,
            use_crowded=self.use_crowded, return_crowded=self.return_crowded,
            return_area=self.return_area
        )

    @attr.slow
    def test_coco_bbox_dataset(self):
        assert_is_bbox_dataset(
            self.dataset, len(coco_bbox_label_names), n_example=30)

        if self.return_crowded:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                _, bbox, _, crowded = self.dataset[i][:4]
                self.assertIsInstance(crowded, np.ndarray)
                self.assertEqual(crowded.dtype, np.bool)
                self.assertEqual(crowded.shape, (bbox.shape[0],))

                if not self.use_crowded:
                    np.testing.assert_equal(crowded, 0)

        if self.return_area:
            for _ in range(10):
                i = np.random.randint(0, len(self.dataset))
                example = self.dataset[i]
                area = example[-1]
                bbox = example[1]
                self.assertIsInstance(area, np.ndarray)
                self.assertEqual(area.dtype, np.float32)


testing.run_module(__name__, __file__)
