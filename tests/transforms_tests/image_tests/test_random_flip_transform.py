import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_flip


class TestRandomFlip(unittest.TestCase):

    def test_random_flip(self):
        img = np.random.uniform(size=(3, 24, 24))

        out, flip_x, flip_y = random_flip(img, random_x=True, random_y=True,
                                          return_flip=True)

        expected = img
        if flip_x:
            expected = expected[:, :, ::-1]
        if flip_y:
            expected = expected[:, ::-1, :]
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
