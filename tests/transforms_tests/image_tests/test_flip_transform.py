import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import flip


class TestRandomFlip(unittest.TestCase):

    def test_random_flip(self):
        img = np.random.uniform(size=(3, 24, 24))

        out = flip(img, y_flip=True, x_flip=True)

        expected = img
        expected = expected[:, :, ::-1]
        expected = expected[:, ::-1, :]
        np.testing.assert_equal(out, expected)

    def test_random_flip_vertical(self):
        img = np.random.uniform(size=(3, 24, 24))

        out = flip(img, y_flip=True, x_flip=False)

        expected = img
        expected = expected[:, ::-1, :]
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
