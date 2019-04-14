import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_point


class TestResizePoint(unittest.TestCase):

    def test_resize_point_ndarray(self):
        point = np.random.uniform(
            low=0., high=32., size=(3, 12, 2))

        out = resize_point(point, in_size=(16, 32), out_size=(8, 64))
        point[:, :, 0] *= 0.5
        point[:, :, 1] *= 2
        np.testing.assert_equal(out, point)

    def test_resize_point_list(self):
        point = [
            np.random.uniform(low=0., high=32., size=(12, 2)),
            np.random.uniform(low=0., high=32., size=(10, 2))
        ]

        out = resize_point(point, in_size=(16, 32), out_size=(8, 64))
        for i, pnt in enumerate(point):
            pnt[:, 0] *= 0.5
            pnt[:, 1] *= 2
            np.testing.assert_equal(out[i], pnt)


testing.run_module(__name__, __file__)
