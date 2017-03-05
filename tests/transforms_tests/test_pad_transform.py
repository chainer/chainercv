import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import pad


class TestPadTransform(unittest.TestCase):

    def test_pad_transform(self):
        x = np.random.uniform(size=(3, 32, 32))
        bg_value = -1

        out = pad(x, (34, 34), bg_value=bg_value)

        np.testing.assert_array_equal(x, out[:, 1:33, 1:33])
        np.testing.assert_array_equal(bg_value, out[:, 0, 0])


testing.run_module(__name__, __file__)
