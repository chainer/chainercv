import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import center_crop


class TestCenterCrop(unittest.TestCase):

    def test_center_crop(self):
        img = np.random.uniform(size=(3, 48, 32))

        out, param = center_crop(img, (16, 24), return_param=True)
        x_slice = param['x_slice']
        y_slice = param['y_slice']

        np.testing.assert_equal(out, img[:, y_slice, x_slice])
        self.assertEqual(x_slice, slice(8, 24))
        self.assertEqual(y_slice, slice(12, 36))


testing.run_module(__name__, __file__)
