from __future__ import division

import unittest

import math
import numpy as np

from chainer import testing
from chainercv.transforms import random_sized_crop


@testing.parameterize(
    {'H': 256, 'W': 256},
    {'H': 129, 'W': 352},
    {'H': 352, 'W': 129},
    {'H': 35, 'W': 500},
)
class TestRandomSizedCrop(unittest.TestCase):

    def test_random_sized_crop(self):
        img = np.random.uniform(size=(3, self.H, self.W))
        scale_ratio_interval = (0.08, 1)
        aspect_ratio_interval = (3 / 4, 4 / 3)
        out, params = random_sized_crop(img, scale_ratio_interval,
                                        aspect_ratio_interval,
                                        return_param=True)

        expected = img[:, params['y_slice'], params['x_slice']]
        np.testing.assert_equal(out, expected)

        _, H_crop, W_crop = out.shape
        scale_ratio = params['scale_ratio']
        aspect_ratio = params['aspect_ratio']
        area = scale_ratio * self.H * self.W
        expected_H_crop = int(math.floor(
            np.sqrt(area * aspect_ratio)))
        expected_W_crop = int(math.floor(
            np.sqrt(area / aspect_ratio)))
        self.assertEqual(H_crop, expected_H_crop)
        self.assertEqual(W_crop, expected_W_crop)

        self.assertTrue(
            (aspect_ratio_interval[0] <= aspect_ratio) and
            (aspect_ratio <= aspect_ratio_interval[1]))
        self.assertTrue(
            scale_ratio <= scale_ratio_interval[1])
        scale_ratio_max = min((scale_ratio_interval[1],
                               self.H / (self.W * aspect_ratio),
                               (aspect_ratio * self.W) / self.H))
        self.assertTrue(
            min((scale_ratio_max, scale_ratio_interval[0])) <= scale_ratio)


testing.run_module(__name__, __file__)
