import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import flip_point


class TestFlipPoint(unittest.TestCase):

    def test_flip_point(self):
        point = np.random.uniform(
            low=0., high=32., size=(12, 2))

        out = flip_point(point, size=(34, 32), y_flip=True)
        point_expected = point.copy()
        point_expected[:, 0] = 34 - point[:, 0]
        np.testing.assert_equal(out, point_expected)

        out = flip_point(point, size=(34, 32), x_flip=True)
        point_expected = point.copy()
        point_expected[:, 1] = 32 - point[:, 1]
        np.testing.assert_equal(out, point_expected)


testing.run_module(__name__, __file__)
