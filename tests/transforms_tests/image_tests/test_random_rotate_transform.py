import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_rotate


class TestRandomRotateTransform(unittest.TestCase):

    def test_random_rotate(self):
        x = np.random.uniform(size=(3, 24, 24))

        out, rotation = random_rotate(x, return_rotation=True)

        expected = np.transpose(x, axes=(1, 2, 0))
        expected = np.rot90(expected, rotation)
        expected = np.transpose(expected, axes=(2, 0, 1))

        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
