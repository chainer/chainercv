import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import pca_lighting


class TestPCALighting(unittest.TestCase):

    def test_pca_lighting(self):
        img = np.random.uniform(size=(3, 48, 32))

        out = pca_lighting(img, 0.1)
        self.assertEqual(img.shape, out.shape)
        self.assertEqual(img.dtype, out.dtype)

        out = pca_lighting(img, 0)
        self.assertEqual(img.shape, out.shape)
        self.assertEqual(img.dtype, out.dtype)
        np.testing.assert_equal(out, img)


testing.run_module(__name__, __file__)
