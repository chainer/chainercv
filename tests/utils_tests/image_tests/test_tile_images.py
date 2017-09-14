from __future__ import division
import math
import numpy as np
import unittest

from chainer import testing

from chainercv.utils import tile_images


@testing.parameterize(*testing.product({
    'fill': [128, (104, 117, 123), np.random.uniform(255, size=(3, 1, 1))],
    'pad': [1, 2, 3]
}))
class TestTileImages(unittest.TestCase):

    def test_tile_images(self):
        B = np.random.randint(10, 20)
        n_col = np.random.randint(2, 5)
        H = 30
        W = 40

        imgs = np.random.uniform(255, size=(B, 3, H, W))
        tile = tile_images(imgs, n_col, self.pad, fill=self.fill)

        n_row = int(math.ceil(B / n_col))
        self.assertTrue(n_col >= 1 and n_row >= 1)
        start_y_11 = H + self.pad + self.pad // 2 + 1
        start_x_11 = W + self.pad + self.pad // 2 + 1
        tile_11 = tile[:,
                       start_y_11:start_y_11 + H,
                       start_x_11:start_x_11 + W]

        np.testing.assert_equal(tile_11, imgs[(n_col - 1) + 2])


testing.run_module(__name__, __file__)
