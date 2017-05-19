import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_crop


class TestRandomCrop(unittest.TestCase):

    def test_random_crop(self):
        img = np.random.uniform(size=(3, 48, 32))

        out, param = random_crop(img, (32, 48), return_param=True)
        x_slice = param['x_slice']
        y_slice = param['y_slice']
        np.testing.assert_equal(out, img)
        self.assertEqual(x_slice, slice(0, 32))
        self.assertEqual(y_slice, slice(0, 48))

        out = random_crop(img, (12, 24))
        self.assertEqual(out.shape[1:], (24, 12))


testing.run_module(__name__, __file__)
