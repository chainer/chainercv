import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_crop


class TestRandomCrop(unittest.TestCase):

    def test_random_crop(self):
        img = np.random.uniform(size=(3, 48, 32))

        out, slices = random_crop(img, (48, 32), return_slices=True)
        np.testing.assert_equal(out, img)
        self.assertEqual(slices[0], slice(0, 48))
        self.assertEqual(slices[1], slice(0, 32))

        out = random_crop(img, (24, 12))
        self.assertEqual(out.shape[1:], (24, 12))


testing.run_module(__name__, __file__)
