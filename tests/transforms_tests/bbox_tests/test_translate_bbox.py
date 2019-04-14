import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import translate_bbox
from chainercv.utils.testing.generate_random_bbox import generate_random_bbox


class TestTranslateBbox(unittest.TestCase):

    def test_translate_bbox(self):
        size = (32, 24)
        y_offset, x_offset = 5, 3
        bbox = generate_random_bbox(10, size, 0, min(size))

        out = translate_bbox(bbox, y_offset=y_offset, x_offset=x_offset)
        bbox_expected = np.empty_like(bbox)
        bbox_expected[:, 0] = bbox[:, 0] + y_offset
        bbox_expected[:, 1] = bbox[:, 1] + x_offset
        bbox_expected[:, 2] = bbox[:, 2] + y_offset
        bbox_expected[:, 3] = bbox[:, 3] + x_offset
        np.testing.assert_equal(out, bbox_expected)


testing.run_module(__name__, __file__)
