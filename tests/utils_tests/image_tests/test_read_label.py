import numpy as np
import tempfile
import unittest

from chainer import testing

from chainercv.utils import read_label
from chainercv.utils import write_image


@testing.parameterize(*testing.product({
    'file_obj': [False, True],
    'format': ['bmp', 'jpeg', 'png'],
    'size': [(48, 32)],
    'dtype': [np.float32, np.uint8, np.int32, bool],
}))
class TestReadLabel(unittest.TestCase):

    def setUp(self):
        if self.file_obj:
            self.f = tempfile.NamedTemporaryFile(delete=False)
            self.file = self.f
            format = self.format
        else:
            if self.format == 'jpeg':
                suffix = '.jpg'
            else:
                suffix = '.' + self.format
            self.f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            self.file = self.f.name
            format = None

        self.img = np.random.randint(
            0, 255, size=self.size, dtype=np.uint8)
        write_image(self.img[None], self.file, format=format)

    def test_read_label(self):
        if self.dtype == np.int32:
            img = read_label(self.file)
        else:
            img = read_label(self.file, dtype=self.dtype)

        self.assertEqual(img.shape, self.size)
        self.assertEqual(img.dtype, self.dtype)

        if self.format in {'bmp', 'png'}:
            np.testing.assert_equal(img, self.img.astype(self.dtype))

    def test_read_label_mutable(self):
        img = read_label(self.file, dtype=self.dtype)
        img[:] = 0
        np.testing.assert_equal(img, 0)


testing.run_module(__name__, __file__)
