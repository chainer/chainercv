import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import flip_bbox
from chainercv.utils.testing.generate_random_bbox import generate_random_bbox


class TestFlipBbox(unittest.TestCase):

    def test_flip_bbox(self):
        bbox = generate_random_bbox(10, (32, 32), 0, 32)

        out = flip_bbox(bbox, size=(34, 32), y_flip=True)
        bbox_expected = bbox.copy()
        bbox_expected[:, 0] = 34 - bbox[:, 2]
        bbox_expected[:, 2] = 34 - bbox[:, 0]
        np.testing.assert_equal(out, bbox_expected)

        out = flip_bbox(bbox, size=(34, 32), x_flip=True)
        bbox_expected = bbox.copy()
        bbox_expected[:, 1] = 32 - bbox[:, 3]
        bbox_expected[:, 3] = 32 - bbox[:, 1]
        np.testing.assert_equal(out, bbox_expected)


testing.run_module(__name__, __file__)
