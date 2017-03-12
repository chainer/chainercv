import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import scale


class TestScale(unittest.TestCase):

    def test_scale_1(self):
        img = np.random.uniform(size=(3, 24, 16))

        out = scale(img, 8)
        self.assertEqual(out.shape, (3, 12, 8))

    def test_scale_2(self):
        img = np.random.uniform(size=(3, 16, 24))

        out = scale(img, 8)
        self.assertEqual(out.shape, (3, 8, 12))


testing.run_module(__name__, __file__)
