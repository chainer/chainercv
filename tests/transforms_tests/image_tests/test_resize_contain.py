import unittest

import numpy as np
import PIL

from chainercv.transforms import resize_contain
from chainercv.utils import testing


@testing.parameterize(*testing.product_dict(
    [
        {'size': (48, 96), 'scaled_size': (32, 64),
         'y_offset': 8, 'x_offset': 16},
        {'size': (16, 68), 'scaled_size': (16, 32),
         'y_offset': 0, 'x_offset': 18},
        {'size': (24, 16), 'scaled_size': (8, 16),
         'y_offset': 8, 'x_offset': 0},
        {'size': (47, 97), 'scaled_size': (32, 64),
         'y_offset': 7, 'x_offset': 16},
    ],
    [
        {'fill': 128},
        {'fill': (104, 117, 123)},
        {'fill': np.random.uniform(255, size=(3, 1, 1))},
    ],
    [
        {'interpolation': PIL.Image.NEAREST},
        {'interpolation': PIL.Image.BILINEAR},
        {'interpolation': PIL.Image.BICUBIC},
        {'interpolation': PIL.Image.LANCZOS},
    ]
))
class TestResizeContain(unittest.TestCase):

    def test_resize_contain(self):
        H, W = 32, 64
        img = np.random.uniform(255, size=(3, H, W))\

        out, param = resize_contain(
            img, self.size, fill=self.fill,
            interpolation=self.interpolation,
            return_param=True)

        self.assertEqual(param['scaled_size'], self.scaled_size)
        self.assertEqual(param['y_offset'], self.y_offset)
        self.assertEqual(param['x_offset'], self.x_offset)

        if self.scaled_size == (H, W):
            np.testing.assert_array_equal(
                out[:,
                    self.y_offset:self.y_offset + H,
                    self.x_offset:self.x_offset + W],
                img)

        if self.y_offset > 0 or self.x_offset > 0:
            if isinstance(self.fill, int):
                fill = (self.fill,) * 3
            else:
                fill = self.fill
            np.testing.assert_array_equal(
                out[:, 0, 0], np.array(fill).flatten())


testing.run_module(__name__, __file__)
