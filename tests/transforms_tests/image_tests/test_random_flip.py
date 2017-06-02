import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_flip


class TestRandomFlip(unittest.TestCase):

    def test_random_flip(self):
        img = np.random.uniform(size=(3, 24, 24))

        out, param = random_flip(
            img, x_random=True, y_random=True, return_param=True)
        x_flip = param['x_flip']
        y_flip = param['y_flip']

        expected = img
        if x_flip:
            expected = expected[:, :, ::-1]
        if y_flip:
            expected = expected[:, ::-1, :]
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
