import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_bbox


class TestResizeBbox(unittest.TestCase):

    def test_resize_bbox(self):
        bboxes = np.random.uniform(
            low=0., high=32., size=(10, 5))

        out = resize_bbox(bboxes, input_shape=(32, 32), output_shape=(64, 128))
        bboxes_expected = bboxes.copy()
        bboxes_expected[:, 0] = bboxes[:, 0] * 4
        bboxes_expected[:, 1] = bboxes[:, 1] * 2
        bboxes_expected[:, 2] = bboxes[:, 2] * 4
        bboxes_expected[:, 3] = bboxes[:, 3] * 2
        np.testing.assert_equal(out, bboxes_expected)


testing.run_module(__name__, __file__)
