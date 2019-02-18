import numpy as np
import unittest


from chainer import testing
from chainercv.transforms import crop_bbox


class TestCropBbox(unittest.TestCase):

    def setUp(self):
        self.bbox = np.array((
            (0, 0, 3, 4),
            (0, 0, 5, 6),
            (0, 5, 3, 6),
            (1, 2, 3, 4),
            (3, 3, 4, 6),
        ), dtype=np.float32)
        self.y_slice = slice(1, 5)
        self.x_slice = slice(0, 4)

    def test_crop_bbox(self):
        expected = np.array((
            (0, 0, 2, 4),
            (0, 0, 4, 4),
            (0, 2, 2, 4),
            (2, 3, 3, 4),
        ), dtype=np.float32)

        out, param = crop_bbox(
            self.bbox, y_slice=self.y_slice, x_slice=self.x_slice,
            return_param=True)
        np.testing.assert_equal(out, expected)
        np.testing.assert_equal(param['index'], (0, 1, 3, 4))
        np.testing.assert_equal(param['truncated_index'], (0, 1, 3))

    def test_crop_bbox_disallow_outside_center(self):
        expected = np.array((
            (0, 0, 2, 4),
            (0, 0, 4, 4),
            (0, 2, 2, 4),
        ), dtype=np.float32)

        out, param = crop_bbox(
            self.bbox, y_slice=self.y_slice, x_slice=self.x_slice,
            allow_outside_center=False, return_param=True)
        np.testing.assert_equal(out, expected)
        np.testing.assert_equal(param['index'], (0, 1, 3))
        np.testing.assert_equal(param['truncated_index'], (0, 1))


testing.run_module(__name__, __file__)
