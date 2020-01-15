from __future__ import division

import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_erasing


@testing.parameterize(*testing.product_dict(
    [
        {'H': 256, 'W': 256},
        {'H': 129, 'W': 352},
        {'H': 352, 'W': 129},
        {'H': 35, 'W': 500},
    ],
    [
        {'prob': 0.0},
        {'prob': 0.5},
        {'prob': 1.0},
    ],
    [
        {'random_value': True},
        {'random_value': False},
    ]
))
class TestRandomErasing(unittest.TestCase):

    def test_random_sized_crop(self):
        img = np.random.uniform(size=(3, self.H, self.W))
        prob = self.prob
        scale_ratio_range = (0.02, 0.4)
        aspect_ratio_range = (0.3, 1 / 0.3)
        fill = np.array((0.4914, 0.4822, 0.4465))
        out, params = random_erasing(img, prob, scale_ratio_range,
                                     aspect_ratio_range, self.random_value,
                                     fill,
                                     return_param=True)

        scale_ratio = params['scale_ratio']
        aspect_ratio = params['aspect_ratio']

        if scale_ratio is not None:
            self.assertTrue(
                (aspect_ratio_range[0] <= aspect_ratio) and
                (aspect_ratio <= aspect_ratio_range[1]))
            self.assertTrue(
                scale_ratio <= scale_ratio_range[1])
            scale_ratio_max = min((scale_ratio_range[1],
                                   self.H / (self.W * aspect_ratio),
                                   (aspect_ratio * self.W) / self.H))
            self.assertTrue(
                min((scale_ratio_max, scale_ratio_range[0])) <= scale_ratio)


testing.run_module(__name__, __file__)
