import numpy as np
import tempfile
import unittest

from PIL import Image

import chainer
from chainer import testing

from chainercv.utils import read_image
from chainercv.utils import write_image

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


def _write_rgba_image(rgba, path):
    rgba = rgba.transpose((1, 2, 0))
    rgba = Image.fromarray(rgba, 'RGBA')
    canvas = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    # Paste the image onto the canvas, using it's alpha channel as mask
    canvas.paste(rgba, mask=rgba)
    canvas.save(path)


def _create_parameters():
    params = testing.product({
        'file_obj': [False, True],
        'size': [(48, 32)],
        'dtype': [np.float32, np.uint8, bool]})
    no_color_params = testing.product({
        'color': [False],
        'alpha': [None],
        'suffix': ['bmp', 'jpg', 'png']})
    no_alpha_params = testing.product({
        'color': [True],
        'alpha': [None],
        'suffix': ['bmp', 'jpg', 'png']})
    alpha_params = testing.product({
        'color': [True],
        'alpha': ['ignore', 'blend_with_white', 'blend_with_black'],
        'suffix': ['png']  # writing alpha image with jpg encoding didn't work
    })
    params = testing.product_dict(
        params,
        no_color_params + no_alpha_params + alpha_params)
    return params


@testing.parameterize(*testing.product_dict(
    _create_parameters(), [{'backend': 'cv2'}, {'backend': 'PIL'}]))
class TestReadImage(unittest.TestCase):

    def setUp(self):
        chainer.config.cv_read_image_backend = self.backend

        self.f = tempfile.NamedTemporaryFile(
            suffix='.' + self.suffix, delete=False)
        if self.file_obj:
            self.file = self.f
        else:
            self.file = self.f.name

        if self.alpha is None:
            if self.color:
                self.img = np.random.randint(
                    0, 255, size=(3,) + self.size, dtype=np.uint8)
            else:
                self.img = np.random.randint(
                    0, 255, size=(1,) + self.size, dtype=np.uint8)
            write_image(self.img, self.f.name)
        else:
            self.img = np.random.randint(
                0, 255, size=(4,) + self.size, dtype=np.uint8)
            _write_rgba_image(self.img, self.f.name)

    def test_read_image_as_color(self):
        img = read_image(self.file, dtype=self.dtype, alpha=self.alpha)

        self.assertEqual(img.shape, (3,) + self.size)
        self.assertEqual(img.dtype, self.dtype)

        if self.suffix in {'bmp', 'png'} and self.alpha is None:
            np.testing.assert_equal(
                img,
                np.broadcast_to(self.img, (3,) + self.size).astype(self.dtype))

    def test_read_image_as_grayscale(self):
        img = read_image(
            self.file, dtype=self.dtype, color=False, alpha=self.alpha)

        self.assertEqual(img.shape, (1,) + self.size)
        self.assertEqual(img.dtype, self.dtype)

        if (self.suffix in {'bmp', 'png'}
                and not self.color and self.alpha is None):
            np.testing.assert_equal(img, self.img.astype(self.dtype))

    def test_read_image_mutable(self):
        img = read_image(self.file, dtype=self.dtype, alpha=self.alpha)
        img[:] = 0
        np.testing.assert_equal(img, 0)


@testing.parameterize(*_create_parameters())
class TestReadImageDifferentBackends(unittest.TestCase):

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(
            suffix='.' + self.suffix, delete=False)
        if self.file_obj:
            self.file = self.f
        else:
            self.file = self.f.name

        if self.alpha is None:
            if self.color:
                self.img = np.random.randint(
                    0, 255, size=(3,) + self.size, dtype=np.uint8)
            else:
                self.img = np.random.randint(
                    0, 255, size=(1,) + self.size, dtype=np.uint8)
            write_image(self.img, self.f.name)
        else:
            self.img = np.random.randint(
                0, 255, size=(4,) + self.size, dtype=np.uint8)
            _write_rgba_image(self.img, self.f.name)

    @unittest.skipUnless(_cv2_available, 'cv2 is not installed')
    def test_read_image_different_backends_as_color(self):
        chainer.config.cv_read_image_backend = 'cv2'
        cv2_img = read_image(
            self.file, dtype=self.dtype, color=self.color, alpha=self.alpha)

        chainer.config.cv_read_image_backend = 'PIL'
        pil_img = read_image(
            self.file, dtype=self.dtype, color=self.color, alpha=self.alpha)

        if self.suffix != 'jpg':
            if self.dtype == np.float32 and self.alpha is not None:
                np.testing.assert_almost_equal(cv2_img, pil_img, decimal=4)
            else:
                np.testing.assert_equal(cv2_img, pil_img)
        else:
            # jpg decoders are differnet, so they produce different results
            assert np.mean(cv2_img == pil_img) > 0.99


testing.run_module(__name__, __file__)
