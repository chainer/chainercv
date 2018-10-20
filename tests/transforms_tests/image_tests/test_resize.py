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


def _create_paramters():
    params = [
        {'interpolation': PIL.Image.NEAREST},
        {'interpolation': PIL.Image.BILINEAR},
        {'interpolation': PIL.Image.BICUBIC},
        {'interpolation': PIL.Image.LANCZOS}]

    if _cv2_available:
        backend_params = {'backend': ['cv2', 'PIL']}
    else:
        backend_params = {'backend': ['PIL']}
    params = testing.product_dict(params, testing.product(backend_params))
    return params


@testing.parameterize(*_create_paramters())
class TestResize(unittest.TestCase):

    def test_resize_color(self):
        img = np.random.uniform(size=(3, 24, 32))
        chainer.config.cv_resize_backend = self.backend
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (3, 32, 64))

    def test_resize_grayscale(self):
        img = np.random.uniform(size=(1, 24, 32))
        chainer.config.cv_resize_backend = self.backend
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (1, 32, 64))


testing.run_module(__name__, __file__)
