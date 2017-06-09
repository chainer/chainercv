import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import translate_bbox


class TestTranslateBbox(unittest.TestCase):

    def test_translate_bbox(self):
        bbox = np.random.uniform(
            low=0., high=32., size=(10, 4))

        out = translate_bbox(bbox, y_offset=5, x_offset=3)
        bbox_expected = np.empty_like(bbox)
        bbox_expected[:, 0] = bbox[:, 0] + 5
        bbox_expected[:, 1] = bbox[:, 1] + 3
        bbox_expected[:, 2] = bbox[:, 2] + 5
        bbox_expected[:, 3] = bbox[:, 3] + 3
        np.testing.assert_equal(out, bbox_expected)


testing.run_module(__name__, __file__)
