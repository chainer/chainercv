import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import flip_bbox


class TestFlipBbox(unittest.TestCase):

    def test_flip_bbox(self):
        bbox = np.random.uniform(
            low=0., high=32., size=(10, 4))

        out = flip_bbox(bbox, img_shape=(32, 32),
                        h_flip=True)
        bbox_expected = bbox.copy()
        bbox_expected[:, 0] = 31 - bbox[:, 2]
        bbox_expected[:, 2] = 31 - bbox[:, 0]
        np.testing.assert_equal(out, bbox_expected)

        out = flip_bbox(bbox, img_shape=(32, 32),
                        v_flip=True)
        bbox_expected = bbox.copy()
        bbox_expected[:, 1] = 31 - bbox[:, 3]
        bbox_expected[:, 3] = 31 - bbox[:, 1]
        np.testing.assert_equal(out, bbox_expected)


testing.run_module(__name__, __file__)
