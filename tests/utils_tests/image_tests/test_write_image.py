import numpy as np
from PIL import Image
import tempfile
import unittest

from chainer import testing

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
            shape = (3,) + self.size
        else:
            shape = (1,) + self.size

        self.img = np.random.randint(0, 255, size=shape).astype(self.dtype)

    def test_write_image(self):
        write_image(self.img, self.path)

        img = Image.open(self.path)

        W, H = img.size
        self.assertEqual((H, W), self.size)

        if self.color:
            self.assertEqual(len(img.getbands()), 3)
        else:
            self.assertEqual(len(img.getbands()), 1)

        if self.suffix in {'bmp', 'png'}:
            img = np.asarray(img)

            if img.ndim == 2:
                # reshape (H, W) -> (1, H, W)
                img = img[np.newaxis]
            else:
                # transpose (H, W, C) -> (C, H, W)
                img = img.transpose((2, 0, 1))

            np.testing.assert_equal(img, self.img)


testing.run_module(__name__, __file__)
