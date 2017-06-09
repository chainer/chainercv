import unittest

import numpy as np
import PIL

from chainer import testing
from chainercv.transforms import resize


@testing.parameterize(
    {'interpolation': PIL.Image.NEAREST},
    {'interpolation': PIL.Image.BILINEAR},
    {'interpolation': PIL.Image.BICUBIC},
    {'interpolation': PIL.Image.LANCZOS},
)
class TestResize(unittest.TestCase):

    def test_resize_color(self):
        img = np.random.uniform(size=(3, 24, 32))
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (3, 32, 64))

    def test_resize_grayscale(self):
        img = np.random.uniform(size=(1, 24, 32))
        out = resize(img, size=(32, 64), interpolation=self.interpolation)
        self.assertEqual(out.shape, (1, 32, 64))


testing.run_module(__name__, __file__)
