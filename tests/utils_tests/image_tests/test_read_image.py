import numpy as np
from PIL import Image
import tempfile
import unittest

from chainer import testing

from chainercv.utils import read_image


@testing.parameterize(*testing.product({
    'size': [(48, 32)],
    'color': [True, False],
    'suffix': ['bmp', 'jpg', 'png'],
    'dtype': [np.float32, np.uint8, bool],
}))
class TestReadImage(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(
            suffix='.' + self.suffix, delete=False)
        self.path = self.file.name

        if self.color:
            self.img = np.random.randint(
                0, 255, size=(3,) + self.size, dtype=np.uint8)
            Image.fromarray(self.img.transpose(1, 2, 0)).save(self.path)
        else:
            self.img = np.random.randint(
                0, 255, size=(1,) + self.size, dtype=np.uint8)
            Image.fromarray(self.img[0]).save(self.path)

    def test_read_image_as_color(self):
        if self.dtype == np.float32:
            img = read_image(self.path)
        else:
            img = read_image(self.path, dtype=self.dtype)

        self.assertEqual(img.shape, (3,) + self.size)
        self.assertEqual(img.dtype, self.dtype)

        if self.suffix in {'bmp', 'png'}:
            np.testing.assert_equal(
                img,
                np.broadcast_to(self.img, (3,) + self.size).astype(self.dtype))

    def test_read_image_as_grayscale(self):
        if self.dtype == np.float32:
            img = read_image(self.path, color=False)
        else:
            img = read_image(self.path, dtype=self.dtype, color=False)

        self.assertEqual(img.shape, (1,) + self.size)
        self.assertEqual(img.dtype, self.dtype)

        if self.suffix in {'bmp', 'png'} and not self.color:
            np.testing.assert_equal(img, self.img.astype(self.dtype))

    def test_read_image_mutable(self):
        img = read_image(self.path)
        img[:] = 0
        np.testing.assert_equal(img, 0)


testing.run_module(__name__, __file__)
