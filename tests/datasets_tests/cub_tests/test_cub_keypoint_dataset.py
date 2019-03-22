import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import CUBKeypointDataset
from chainercv.utils import assert_is_bbox
from chainercv.utils import assert_is_point_dataset


@testing.parameterize(*testing.product({
    'return_bbox': [True, False],
    'return_prob_map': [True, False]}
))
class TestCUBKeypointDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = CUBKeypointDataset(return_bbox=self.return_bbox,
                                          return_prob_map=self.return_prob_map)

    @attr.slow
    def test_cub_point_dataset(self):
        assert_is_point_dataset(
            self.dataset, n_point=15, n_example=10)

        idx = np.random.choice(np.arange(10))
        if self.return_bbox:
            if self.return_prob_map:
                bbox = self.dataset[idx][-2]
            else:
                bbox = self.dataset[idx][-1]
            assert_is_bbox(bbox)
        if self.return_prob_map:
            img = self.dataset[idx][0]
            prob_map = self.dataset[idx][-1]
            self.assertEqual(prob_map.dtype, np.float32)
            self.assertEqual(prob_map.shape, img.shape[1:])
            self.assertTrue(np.min(prob_map) >= 0)
            self.assertTrue(np.max(prob_map) <= 1)


testing.run_module(__name__, __file__)
