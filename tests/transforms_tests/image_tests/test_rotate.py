import PIL
import random
import unittest

import numpy as np

import chainer
from chainer import testing
from chainercv.transforms import flip
from chainercv.transforms import rotate

try:
    import scipy  # NOQA
    _available = True
except ImportError:
    _available = False


@testing.parameterize(*testing.product({
    'interpolation': [PIL.Image.NEAREST, PIL.Image.BILINEAR,
                      PIL.Image.BICUBIC],
    'fill': [-1, 0, 100],
    'size': [(3, 32, 24), (1, 32, 24)],
}))
class TestRotate(unittest.TestCase):

    def test_rotate_pil(self):
        chainer.global_config.cv_rotate_backend = 'PIL'
        img = np.random.uniform(0, 256, size=self.size).astype(np.float32)
        angle = random.uniform(-180, 180)

        out = rotate(img, angle, fill=self.fill,
                     interpolation=self.interpolation)
        expected = flip(img, x_flip=True)
        expected = rotate(
            expected, -1 * angle, fill=self.fill,
            interpolation=self.interpolation)
        expected = flip(expected, x_flip=True)

        if self.interpolation == PIL.Image.NEAREST:
            assert np.mean(out == expected) > 0.99
        else:
            np.testing.assert_almost_equal(out, expected, decimal=3)

    def test_rotate_cv2(self):
        chainer.global_config.cv_rotate_backend = 'cv2'
        img = np.random.uniform(0, 256, size=self.size).astype(np.float32)
        angle = random.uniform(-180, 180)

        out = rotate(img, angle, fill=self.fill,
                     interpolation=self.interpolation)
        opposite_out = rotate(img, -angle, fill=self.fill,
                              interpolation=self.interpolation)

        assert out.shape[1:] == opposite_out.shape[1:]

    def test_rotate_pil_no_expand(self):
        chainer.global_config.cv_rotate_backend = 'PIL'
        img = np.random.uniform(0, 256, size=self.size).astype(np.float32)
        angle = random.uniform(-180, 180)

        out = rotate(img, angle, fill=self.fill,
                     expand=False,
                     interpolation=self.interpolation)
        assert out.shape == img.shape

    def test_rotate_cv2_no_expand(self):
        chainer.global_config.cv_rotate_backend = 'cv2'
        img = np.random.uniform(0, 256, size=self.size).astype(np.float32)
        angle = random.uniform(-180, 180)

        out = rotate(img, angle, fill=self.fill,
                     expand=False,
                     interpolation=self.interpolation)
        assert out.shape == img.shape


testing.run_module(__name__, __file__)
