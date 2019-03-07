import numpy as np
from PIL import Image
import tempfile
import unittest

from chainer import testing

from chainercv.utils import write_image


@testing.parameterize(*testing.product({
    'file_obj': [False, True],
    'format': ['bmp', 'jpeg', 'png'],
    'size': [(48, 32)],
    'color': [True, False],
    'dtype': [np.float32, np.uint8, bool],
}))
class TestWriteImage(unittest.TestCase):

    def setUp(self):
        if self.file_obj:
            self.f = tempfile.NamedTemporaryFile(delete=False)
            self.file = self.f
        else:
            if self.format == 'jpeg':
                suffix = '.jpg'
            else:
                suffix = '.' + self.format
            self.f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            self.file = self.f.name

        if self.color:
            shape = (3,) + self.size
        else:
            shape = (1,) + self.size

        self.img = np.random.randint(0, 255, size=shape).astype(self.dtype)

    def test_write_image(self):
        if self.file_obj:
            write_image(self.img, self.file, format=self.format)
        else:
            write_image(self.img, self.file)
        img = Image.open(self.file)

        W, H = img.size
        self.assertEqual((H, W), self.size)

        if self.color:
            self.assertEqual(len(img.getbands()), 3)
        else:
            self.assertEqual(len(img.getbands()), 1)

        if self.format in {'bmp', 'png'}:
            img = np.asarray(img)

            if img.ndim == 2:
                # reshape (H, W) -> (1, H, W)
                img = img[np.newaxis]
            else:
                # transpose (H, W, C) -> (C, H, W)
                img = img.transpose((2, 0, 1))

            np.testing.assert_equal(img, self.img)


testing.run_module(__name__, __file__)
