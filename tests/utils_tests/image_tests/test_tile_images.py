from __future__ import division
import math
import numpy as np
import unittest

from chainercv.utils import testing
from chainercv.utils import tile_images


@testing.parameterize(*testing.product({
    'fill': [128, (104, 117, 123), np.random.uniform(255, size=(3, 1, 1))],
    'pad': [0, 1, 2, 3, (3, 5), (5, 2)]
}))
class TestTileImages(unittest.TestCase):

    def test_tile_images(self):
        B = np.random.randint(10, 20)
        n_col = np.random.randint(2, 5)
        H = 30
        W = 40

        imgs = np.random.uniform(255, size=(B, 3, H, W))
        tile = tile_images(imgs, n_col, self.pad, fill=self.fill)

        if isinstance(self.pad, int):
            pad_y = self.pad
            pad_x = self.pad
        else:
            pad_y, pad_x = self.pad
        n_row = int(math.ceil(B / n_col))
        self.assertTrue(n_col >= 1 and n_row >= 1)
        start_y_11 = H + pad_y + pad_y // 2
        start_x_11 = W + pad_x + pad_x // 2
        tile_11 = tile[:,
                       start_y_11:start_y_11 + H,
                       start_x_11:start_x_11 + W]

        np.testing.assert_equal(tile_11, imgs[(n_col - 1) + 2])


testing.run_module(__name__, __file__)
