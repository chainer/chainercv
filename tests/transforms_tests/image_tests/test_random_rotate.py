import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_rotate


class TestRandomRotate(unittest.TestCase):

    def test_random_rotate(self):
        img = np.random.uniform(size=(3, 24, 24))

        out, param = random_rotate(img, return_param=True)
        k = param['k']

        expected = np.transpose(img, axes=(1, 2, 0))
        expected = np.rot90(expected, k)
        expected = np.transpose(expected, axes=(2, 0, 1))

        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
