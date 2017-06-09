import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_bbox


class TestResizeBbox(unittest.TestCase):

    def test_resize_bbox(self):
        bbox = np.random.uniform(
            low=0., high=32., size=(10, 4))

        out = resize_bbox(bbox, in_size=(32, 32), out_size=(64, 128))
        bbox_expected = bbox.copy()
        bbox_expected[:, 0] = bbox[:, 0] * 2
        bbox_expected[:, 1] = bbox[:, 1] * 4
        bbox_expected[:, 2] = bbox[:, 2] * 2
        bbox_expected[:, 3] = bbox[:, 3] * 4
        np.testing.assert_equal(out, bbox_expected)


testing.run_module(__name__, __file__)
