import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_point


class TestResizePoint(unittest.TestCase):

    def test_resize_point(self):
        point = np.random.uniform(
            low=0., high=32., size=(12, 2))

        out = resize_point(point, in_size=(16, 32), out_size=(8, 64))
        point[:, 0] *= 0.5
        point[:, 1] *= 2
        np.testing.assert_equal(out, point)


testing.run_module(__name__, __file__)
