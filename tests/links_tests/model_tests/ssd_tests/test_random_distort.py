import numpy as np
import unittest

from chainer import testing
from chainercv.links.model.ssd import random_distort

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


@unittest.skipUnless(_cv2_available, 'cv2 is not installed')
class TestRandomDistort(unittest.TestCase):

    def test_random_distort(self):
        img = np.random.randint(0, 256, size=(3, 48, 32)).astype(np.float32)

        out = random_distort(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, img.dtype)


testing.run_module(__name__, __file__)
