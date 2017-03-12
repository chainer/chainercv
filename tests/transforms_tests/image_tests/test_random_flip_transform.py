import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_flip


class TestRandomFlipTransform(unittest.TestCase):

    def test_random_flip(self):
        x = np.random.uniform(size=(3, 24, 24))

        out, flips = random_flip(x, random_h=True, random_v=True,
                                 return_flip=True)

        expected = x
        if flips['h']:
            expected = expected[:, :, ::-1]
        if flips['v']:
            expected = expected[:, ::-1, :]
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
