import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_flip


class TestRandomFlip(unittest.TestCase):

    def test_random_flip(self):
        img = np.random.uniform(size=(3, 24, 24))

        out, x_flip, y_flip = random_flip(img, random_x=True, random_y=True,
                                          return_flip=True)

        expected = img
        if x_flip:
            expected = expected[:, :, ::-1]
        if y_flip:
            expected = expected[:, ::-1, :]
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
