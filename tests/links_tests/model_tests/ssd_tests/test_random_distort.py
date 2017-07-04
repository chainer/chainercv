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

    def test_random_distort(self):
        if not optional_modules:
            return
        img = np.random.randint(0, 256, size=(3, 48, 32)).astype(np.float32)

        out = random_distort(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, img.dtype)


testing.run_module(__name__, __file__)
