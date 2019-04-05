import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import flip_point


class TestFlipPoint(unittest.TestCase):

    def test_flip_point_ndarray(self):
        point = np.random.uniform(
            low=0., high=32., size=(3, 12, 2))

        out = flip_point(point, size=(34, 32), y_flip=True)
        point_expected = point.copy()
        point_expected[:, :, 0] = 34 - point[:, :, 0]
        np.testing.assert_equal(out, point_expected)

        out = flip_point(point, size=(34, 32), x_flip=True)
        point_expected = point.copy()
        point_expected[:, :, 1] = 32 - point[:, :, 1]
        np.testing.assert_equal(out, point_expected)

    def test_flip_point_list(self):
        point = [
            np.random.uniform(low=0., high=32., size=(12, 2)),
            np.random.uniform(low=0., high=32., size=(10, 2)),
        ]

        out = flip_point(point, size=(34, 32), y_flip=True)
        for i, pnt in enumerate(point):
            pnt_expected = pnt.copy()
            pnt_expected[:, 0] = 34 - pnt[:, 0]
            np.testing.assert_equal(out[i], pnt_expected)

        out = flip_point(point, size=(34, 32), x_flip=True)
        for i, pnt in enumerate(point):
            pnt_expected = pnt.copy()
            pnt_expected[:, 1] = 32 - pnt[:, 1]
            np.testing.assert_equal(out[i], pnt_expected)


testing.run_module(__name__, __file__)
