import numpy as np
import unittest

from chainer import testing
from chainercv.links.model.ssd import resize_with_random_interpolation

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


@unittest.skipUnless(_cv2_available, 'cv2 is not installed')
class TestResizeWithRandomInterpolation(unittest.TestCase):

    def test_resize_color(self):
        img = np.random.uniform(size=(3, 24, 32))
        out = resize_with_random_interpolation(img, size=(32, 64))
        self.assertEqual(out.shape, (3, 32, 64))

    def test_resize_grayscale(self):
        img = np.random.uniform(size=(1, 24, 32))
        out = resize_with_random_interpolation(img, size=(32, 64))
        self.assertEqual(out.shape, (1, 32, 64))


testing.run_module(__name__, __file__)
