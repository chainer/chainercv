import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import pad


class TestPadTransform(unittest.TestCase):

    def test_pad_transform(self):
        img = np.random.uniform(size=(3, 32, 32))
        bg_value = -1

        out = pad(img, (34, 36), bg_value=bg_value)

        np.testing.assert_array_equal(img, out[:, 2:34, 1:33])
        np.testing.assert_array_equal(bg_value, out[:, 0, 0])


testing.run_module(__name__, __file__)
