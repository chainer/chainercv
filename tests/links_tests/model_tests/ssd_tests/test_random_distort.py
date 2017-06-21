import numpy as np
import unittest

from chainer import testing
from chainercv.links.model.ssd import random_distort

try:
    import cv2  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


class TestRandomDistort(unittest.TestCase):

    def test_pca_lighting(self):
        img = np.random.randint(0, 256, size=(3, 48, 32))

        out = random_distort(img)
        self.assertEqual(img.shape, out.shape)
        self.assertEqual(img.dtype, out.dtype)


testing.run_module(__name__, __file__)
