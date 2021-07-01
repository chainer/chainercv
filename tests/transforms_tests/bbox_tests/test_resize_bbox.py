import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_bbox
from chainercv.utils.testing.generate_random_bbox import generate_random_bbox


class TestResizeBbox(unittest.TestCase):

    def test_resize_bbox(self):
        in_size = (32, 24)
        out_size = (in_size[0] * 2, in_size[1] * 4)
        bbox = generate_random_bbox(10, in_size, 0, min(in_size))

        out = resize_bbox(bbox, in_size=in_size, out_size=out_size)
        bbox_expected = bbox.copy()
        bbox_expected[:, 0] = bbox[:, 0] * 2
        bbox_expected[:, 1] = bbox[:, 1] * 4
        bbox_expected[:, 2] = bbox[:, 2] * 2
        bbox_expected[:, 3] = bbox[:, 3] * 4
        np.testing.assert_equal(out, bbox_expected)


testing.run_module(__name__, __file__)
