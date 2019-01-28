import numpy as np
import tempfile
import unittest

from chainer import testing

from chainercv.utils import read_label
from chainercv.utils import write_image


@testing.parameterize(*testing.product({
    'size': [(48, 32)],
    'suffix': ['bmp', 'jpg', 'png'],
    'dtype': [np.float32, np.uint8, bool],
}))
class TestReadImage(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(
            suffix='.' + self.suffix, delete=False)
        self.path = self.file.name

        self.img = np.random.randint(
            0, 255, size=(1,) + self.size, dtype=np.uint8)
        write_image(self.img, self.path)

    def test_read_label(self):
        if self.dtype == np.uint8:
            img = read_label(self.path)
        else:
            img = read_label(self.path, dtype=self.dtype)

        self.assertEqual(img.shape, (1,) + self.size)
        self.assertEqual(img.dtype, self.dtype)

        print(self.img.astype(self.dtype))
        print(img)

        if self.suffix in {'bmp', 'png'}:
            np.testing.assert_equal(img, self.img.astype(self.dtype))

    def test_read_label_mutable(self):
        img = read_label(self.path, dtype=self.dtype)
        img[:] = 0
        np.testing.assert_equal(img, 0)


testing.run_module(__name__, __file__)
