import unittest

import numpy as np
import PIL

import chainer
from chainer.cuda import to_cpu
from chainer.cuda import to_gpu
from chainer import testing
from chainer.testing import attr
from chainercv.transforms import resize


@testing.parameterize(
    {'interpolation': PIL.Image.NEAREST},
    {'interpolation': PIL.Image.BILINEAR},
    {'interpolation': PIL.Image.BICUBIC},
    {'interpolation': PIL.Image.LANCZOS},
)
class TestResize(unittest.TestCase):

    def check_resize_color(self, xp):
        img = xp.random.uniform(size=(3, 24, 32)).astype(np.float32)
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (3, 32, 64))

    def test_resize_color_cpu(self):
        self.check_resize_color(np)

    @attr.gpu
    def test_resize_color_gpu(self):
        if self.interpolation == PIL.Image.BILINEAR:
            self.check_resize_color(chainer.cuda.cupy)

    def check_resize_grayscale(self, xp):
        img = xp.random.uniform(size=(1, 24, 32)).astype(np.float32)
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (1, 32, 64))

    def test_resize_grayscale_cpu(self):
        self.check_resize_grayscale(np)

    @attr.gpu
    def test_resize_grayscale_gpu(self):
        if self.interpolation == PIL.Image.BILINEAR:
            self.check_resize_grayscale(chainer.cuda.cupy)


@testing.parameterize(
    {'n_channel': 1},
    {'n_channel': 3},
)
class TestResizeDownscale(unittest.TestCase):

    in_size = (4, 4)
    out_size = (2, 2)

    def setUp(self):
        self.x = np.zeros((self.n_channel,) + self.in_size, dtype=np.float32)
        self.x[:, :2, :2] = 1
        self.x[:, 2:, :2] = 2
        self.x[:, :2, 2:] = 3
        self.x[:, 2:, 2:] = 4

        self.out = np.zeros(
            (self.n_channel,) + self.out_size, dtype=np.float32)
        self.out[:, 0, 0] = 1
        self.out[:, 1, 0] = 2
        self.out[:, 0, 1] = 3
        self.out[:, 1, 1] = 4

    def check_downscale(self, x, size):
        y = resize(x, size)
        np.testing.assert_allclose(
            to_cpu(y), self.out)

    def test_downscale_cpu(self):
        self.check_downscale(self.x, self.out_size)

    @attr.gpu
    def test_downscale_gpu(self):
        self.check_downscale(to_gpu(self.x), self.out_size)


@testing.parameterize(
    {'n_channel': 1},
    {'n_channel': 3},
)
class TestResizeUpscale(unittest.TestCase):

    in_size = (2, 2)
    out_size = (3, 3)

    def setUp(self):
        self.x = np.zeros((self.n_channel,) + self.in_size, dtype=np.float32)
        self.x[:, 0, 0] = 1
        self.x[:, 1, 0] = 2
        self.x[:, 0, 1] = 3
        self.x[:, 1, 1] = 4

        self.out = np.zeros(
            (self.n_channel,) + self.out_size, dtype=np.float32)
        self.out[:, :, :] = np.array([[1., 2., 3.],
                                      [1.5, 2.5, 3.5],
                                      [2., 3., 4.]],
                                     dtype=np.float32)

    def check_upscale(self, x, size):
        y = resize(x, size)
        np.testing.assert_allclose(to_cpu(y), self.out)

    def test_upscale(self):
        self.check_upscale(self.x, self.out_size)

    @attr.gpu
    def test_upscale_gpu(self):
        self.check_upscale(to_gpu(self.x), self.out_size)


testing.run_module(__name__, __file__)
