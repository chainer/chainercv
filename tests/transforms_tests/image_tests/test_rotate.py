import PIL
import random
import unittest

import numpy as np

import chainer
from chainer import testing
from chainercv.transforms import flip
from chainercv.transforms import rotate

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


@testing.parameterize(*testing.product({
    'interpolation': [PIL.Image.NEAREST, PIL.Image.BILINEAR,
                      PIL.Image.BICUBIC],
    'fill': [-1, 0, 100],
    'size': [(3, 32, 24), (1, 32, 24)],
}))
class TestRotate(unittest.TestCase):

    def test_rotate_pil(self):
        img = np.random.uniform(0, 256, size=self.size).astype(np.float32)
        angle = random.uniform(-180, 180)

        with chainer.using_config('cv_rotate_backend', 'PIL'):
            out = rotate(img, angle, fill=self.fill,
                         interpolation=self.interpolation)
        expected = flip(img, x_flip=True)
        with chainer.using_config('cv_rotate_backend', 'PIL'):
            expected = rotate(
                expected, -1 * angle, fill=self.fill,
                interpolation=self.interpolation)
        expected = flip(expected, x_flip=True)

        if self.interpolation == PIL.Image.NEAREST:
            assert np.mean(out == expected) > 0.99
        else:
            np.testing.assert_almost_equal(out, expected, decimal=3)

    def test_rotate_none_and_cv2(self):
        backends = [None, 'cv2'] if _cv2_available else [None]
        for backend in backends:
            img = np.random.uniform(0, 256, size=self.size).astype(np.float32)
            angle = random.uniform(-180, 180)

            with chainer.using_config('cv_rotate_backend', backend):
                out = rotate(img, angle, fill=self.fill,
                             interpolation=self.interpolation)
                opposite_out = rotate(img, -angle, fill=self.fill,
                                      interpolation=self.interpolation)

            assert out.shape[1:] == opposite_out.shape[1:]

    def test_rotate_no_expand(self):
        backends = [None, 'cv2', 'PIL'] if _cv2_available else [None, 'PIL']
        for backend in backends:
            img = np.random.uniform(0, 256, size=self.size).astype(np.float32)
            angle = random.uniform(-180, 180)

            with chainer.using_config('cv_rotate_backend', backend):
                out = rotate(img, angle, fill=self.fill,
                             expand=False,
                             interpolation=self.interpolation)
            assert out.shape == img.shape


@unittest.skipUnless(not _cv2_available, 'cv2 is installed')
class TestRotateRaiseErrorWithCv2(unittest.TestCase):

    def test_rotate_raise_error_with_cv2(self):
        img = np.random.uniform(0, 256, size=(3, 32, 24)).astype(np.float32)
        angle = random.uniform(-180, 180)
        with chainer.using_config('cv_rotate_backend', 'cv2'):
            with self.assertRaises(ValueError):
                rotate(img, angle)


testing.run_module(__name__, __file__)
