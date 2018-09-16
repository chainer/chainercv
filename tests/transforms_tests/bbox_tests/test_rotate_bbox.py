import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import rotate_bbox
from chainercv.utils import generate_random_bbox


@testing.parameterize(*testing.product({
    'angle': [180, 90, 0, -90, -180]
}))
class TestRotateBbox(unittest.TestCase):

    def test_rotate_bbox(self):
        size = (32, 24)
        bbox = generate_random_bbox(10, size, 0, 24)

        out = rotate_bbox(bbox, self.angle, size)
        if self.angle % 180 != 0:
            rotate_size = size[::-1]
        else:
            rotate_size = size
        out = rotate_bbox(out, -1 * self.angle, rotate_size)

        np.testing.assert_almost_equal(out, bbox, decimal=6)


testing.run_module(__name__, __file__)
