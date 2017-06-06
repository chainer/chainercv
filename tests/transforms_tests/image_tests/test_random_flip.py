import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_flip


class TestRandomFlip(unittest.TestCase):

    def test_random_flip(self):
        img = np.random.uniform(size=(3, 24, 24))

        out, param = random_flip(
            img, y_random=True, x_random=True, return_param=True)
        y_flip = param['y_flip']
        x_flip = param['x_flip']

        expected = img
        if y_flip:
            expected = expected[:, ::-1, :]
        if x_flip:
            expected = expected[:, :, ::-1]
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
