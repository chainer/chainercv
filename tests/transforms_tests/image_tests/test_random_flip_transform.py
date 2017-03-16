import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_flip


class TestRandomFlip(unittest.TestCase):

    def test_random_flip(self):
        img = np.random.uniform(size=(3, 24, 24))

        out, flip_h, flip_v = random_flip(img, random_h=True, random_v=True,
                                          return_flip=True)

        expected = img
        if flip_h:
            expected = expected[:, :, ::-1]
        if flip_v:
            expected = expected[:, ::-1, :]
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
