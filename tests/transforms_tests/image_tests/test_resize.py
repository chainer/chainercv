import unittest

import numpy as np
import PIL

import chainer
from chainer import testing
from chainercv.transforms import resize

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


@testing.parameterize(*testing.product({
    'interpolation': [PIL.Image.NEAREST, PIL.Image.BILINEAR,
                      PIL.Image.BICUBIC, PIL.Image.LANCZOS],
    'backend': ['cv2', 'PIL', None],
}))
class TestResize(unittest.TestCase):

    def test_resize_color(self):
        if self.backend == 'cv2' and not _cv2_available:
            return
        img = np.random.uniform(size=(3, 24, 32))
        chainer.config.cv_resize_backend = self.backend
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (3, 32, 64))

    def test_resize_grayscale(self):
        if self.backend == 'cv2' and not _cv2_available:
            return
        img = np.random.uniform(size=(1, 24, 32))
        chainer.config.cv_resize_backend = self.backend
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (1, 32, 64))

    def test_zero_length_img(self):
        if self.backend == 'cv2' and not _cv2_available:
            return
        img = np.random.uniform(size=(0, 24, 32))
        chainer.config.cv_resize_backend = self.backend
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (0, 32, 64))


@unittest.skipUnless(not _cv2_available, 'cv2 is installed')
class TestResizeRaiseErrorWithCv2(unittest.TestCase):

    def test_resize_raise_error_with_cv2(self):
        img = np.random.uniform(size=(3, 24, 32))
        chainer.config.cv_resize_backend = 'cv2'
        with self.assertRaises(ValueError):
            resize(img, size=(32, 64))


testing.run_module(__name__, __file__)
