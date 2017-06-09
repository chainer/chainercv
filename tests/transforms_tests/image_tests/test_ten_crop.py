import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import ten_crop


class TestTenCrop(unittest.TestCase):

    def test_ten_crop(self):
        img = np.random.uniform(size=(3, 48, 32))

        out = ten_crop(img, (48, 32))
        self.assertEqual(out.shape, (10, 3, 48, 32))
        for crop in out[:5]:
            np.testing.assert_equal(crop, img)
        for crop in out[5:]:
            np.testing.assert_equal(crop[:, :, ::-1], img)

        out = ten_crop(img, (24, 12))
        self.assertEqual(out.shape, (10, 3, 24, 12))


testing.run_module(__name__, __file__)
