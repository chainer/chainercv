import numpy as np
import tempfile
import unittest

from chainer import testing

from chainercv.utils import read_image
from chainercv.utils import write_image


@testing.parameterize(*testing.product({
    'size': [(48, 32)],
    'color': [True, False],
    'suffix': ['bmp', 'jpg', 'png'],
    'dtype': [np.float32, np.uint8, bool],
}))
class TestWriteImage(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(
            suffix='.' + self.suffix, delete=False)
        self.path = self.file.name

        if self.color:
            self.img = np.random.randint(
                0, 255, size=(3,) + self.size, dtype=self.dtype)
        else:
            self.img = np.random.randint(
                0, 255, size=(1,) + self.size, dtype=self.dtype)

    def test_write_image(self):
        write_image(self.img, self.path)

        img = read_image(self.path, dtype=self.dtype, color=self.color)
        self.assertEqual(img.shape, self.img.shape)
        self.assertEqual(img.dtype, self.dtype)

        if self.suffix in {'bmp', 'png'}:
            np.testing.assert_equal(img, self.img)


testing.run_module(__name__, __file__)
