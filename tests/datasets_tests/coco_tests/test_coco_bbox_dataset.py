import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.utils import assert_is_bbox_dataset


@testing.parameterize(*testing.product({
    'split': ['train', 'val', 'minival', 'valminusminival'],
    'use_crowded': [False, True],
    'return_crowded': [False, True]
}))
class TestCOCOBboxDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = COCOBboxDataset(
            split=self.split,
            use_crowded=self.use_crowded, return_crowded=self.return_crowded)

    @attr.slow
    def test_as_bbox_dataset(self):
        assert_is_bbox_dataset(
            self.dataset, len(coco_bbox_label_names), n_example=30)

    @attr.slow
    def test_crowded(self):
        if not self.return_crowded:
            return

        for _ in range(10):
            i = np.random.randint(0, len(self.dataset))
            _, bbox, _, crowded = self.dataset[i]
            self.assertIsInstance(crowded, np.ndarray)
            self.assertEqual(crowded.dtype, np.bool)
            self.assertEqual(crowded.shape, (bbox.shape[0],))

            if not self.use_crowded:
                np.testing.assert_equal(crowded, 0)



testing.run_module(__name__, __file__)
