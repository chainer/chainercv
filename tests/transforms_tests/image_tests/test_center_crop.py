import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import center_crop


class TestCenterCrop(unittest.TestCase):

    def test_center_crop(self):
        img = np.random.uniform(size=(3, 48, 32))

        out, slice_H, slice_W = center_crop(img, (24, 16), return_slices=True)

        np.testing.assert_equal(out, img[:, slice_H, slice_W])
        self.assertEqual(slice_H, slice(12, 36))
        self.assertEqual(slice_W, slice(8, 24))


testing.run_module(__name__, __file__)
