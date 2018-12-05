import numpy as np
import tempfile
import unittest

import chainer
from chainer import testing

from chainercv.utils import read_image
from chainercv.utils import write_image

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


@testing.parameterize(*testing.product({
    'size': [(48, 32)],
    'color': [True, False],
    'suffix': ['bmp', 'jpg', 'png'],
    'dtype': [np.float32, np.uint8, bool],
    'backend': ['cv2', 'PIL'],
}))
class TestReadImage(unittest.TestCase):

    def setUp(self):
        chainer.config.cv_read_image_backend = self.backend

        self.file = tempfile.NamedTemporaryFile(
            suffix='.' + self.suffix, delete=False)
        self.path = self.file.name

        if self.color:
            self.img = np.random.randint(
                0, 255, size=(3,) + self.size, dtype=np.uint8)
        else:
            self.img = np.random.randint(
                0, 255, size=(1,) + self.size, dtype=np.uint8)
        write_image(self.img, self.path)

    def test_read_image_as_color(self):
        img = read_image(self.path, dtype=self.dtype)

        self.assertEqual(img.shape, (3,) + self.size)
        self.assertEqual(img.dtype, self.dtype)

        if self.suffix in {'bmp', 'png'}:
            np.testing.assert_equal(
                img,
                np.broadcast_to(self.img, (3,) + self.size).astype(self.dtype))

    def test_read_image_as_grayscale(self):
        img = read_image(self.path, dtype=self.dtype, color=False)

        self.assertEqual(img.shape, (1,) + self.size)
        self.assertEqual(img.dtype, self.dtype)

        if self.suffix in {'bmp', 'png'} and not self.color:
            np.testing.assert_equal(img, self.img.astype(self.dtype))

    def test_read_image_mutable(self):
        img = read_image(self.path)
        img[:] = 0
        np.testing.assert_equal(img, 0)


@testing.parameterize(*testing.product({
    'size': [(48, 32)],
    'color': [True, False],
    'suffix': ['bmp', 'jpg', 'png'],
    'dtype': [np.float32, np.uint8, bool]}))
class TestReadImageDifferentBackends(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(
            suffix='.' + self.suffix, delete=False)
        self.path = self.file.name

        if self.color:
            self.img = np.random.randint(
                0, 255, size=(3,) + self.size, dtype=np.uint8)
        else:
            self.img = np.random.randint(
                0, 255, size=(1,) + self.size, dtype=np.uint8)
        write_image(self.img, self.path)

    @unittest.skipUnless(_cv2_available, 'cv2 is not installed')
    def test_read_image_different_backends_as_color(self):
        chainer.config.cv_read_image_backend = 'cv2'
        cv2_img = read_image(self.path, dtype=self.dtype, color=True)

        chainer.config.cv_read_image_backend = 'PIL'
        pil_img = read_image(self.path, dtype=self.dtype, color=True)

        if self.suffix != 'jpg':
            np.testing.assert_equal(cv2_img, pil_img)
        else:
            # jpg decoders are differnet, so they produce different results
            assert np.mean(cv2_img == pil_img) > 0.99


testing.run_module(__name__, __file__)
