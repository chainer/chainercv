import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import flip_bbox


class TestFlipBboxTransform(unittest.TestCase):

    def test_flip_bbox(self):
        bboxes = np.random.uniform(
            low=0., high=32., size=(10, 5))

        out = flip_bbox(bboxes, img_shape=(32, 32),
                        h_flip=True)
        bboxes_expected = bboxes.copy()
        bboxes_expected[:, 0] = 31 - bboxes[:, 2]
        bboxes_expected[:, 2] = 31 - bboxes[:, 0]
        np.testing.assert_equal(out, bboxes_expected)

        out = flip_bbox(bboxes, img_shape=(32, 32),
                        v_flip=True)
        bboxes_expected = bboxes.copy()
        bboxes_expected[:, 1] = 31 - bboxes[:, 3]
        bboxes_expected[:, 3] = 31 - bboxes[:, 1]
        np.testing.assert_equal(out, bboxes_expected)


testing.run_module(__name__, __file__)
