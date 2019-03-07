import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import translate_point


class TestTranslatePoint(unittest.TestCase):

    def test_translate_point_ndarray(self):
        point = np.random.uniform(
            low=0., high=32., size=(3, 10, 2))

        out = translate_point(point, y_offset=3, x_offset=5)
        expected = np.empty_like(point)
        expected[:, :, 0] = point[:, :, 0] + 3
        expected[:, :, 1] = point[:, :, 1] + 5
        np.testing.assert_equal(out, expected)

    def test_translate_point_list(self):
        point = [np.random.uniform(
            low=0., high=32., size=(10, 2))]

        out = translate_point(point, y_offset=3, x_offset=5)
        for i, pnt in enumerate(point):
            expected = np.empty_like(pnt)
            expected[:, 0] = pnt[:,  0] + 3
            expected[:, 1] = pnt[:,  1] + 5
            np.testing.assert_equal(out[i], expected)


testing.run_module(__name__, __file__)
