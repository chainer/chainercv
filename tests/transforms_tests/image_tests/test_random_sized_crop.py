import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_sized_crop


@testing.parameterize(
    {'H': 256, 'W': 256},
    {'H': 129, 'W': 352},
    {'H': 352, 'W': 129},
)
class TestRandomSizedCrop(unittest.TestCase):

    def test_random_sized_crop(self):
        img = np.random.uniform(size=(3, self.H, self.W))

        out, params = random_sized_crop(img, return_params=True)

        expected = img[:, params['y_slice'], params['x_slice']]
        np.testing.assert_equal(out, expected)

        _, H_crop, W_crop = out.shape
        s = params['scale_ratio']
        a = params['aspect_ratio']
        self.assertEqual(H_crop, int(s * self.H * np.sqrt(a)))
        self.assertEqual(W_crop, int(s * self.W / np.sqrt(a)))


testing.run_module(__name__, __file__)
