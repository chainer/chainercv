from __future__ import division

import math
import numpy as np


def make_grid(imgs, n_col, pad=2, fill=0):
    """Make a grid of images.

    Args:
        imgs (numpy.ndarray): A batch of images whose shape is BCHW.
        n_col (int): Number of columns in a grid.
        pad (int): Amount of pad. Default is 2.
        fill (float, tuple or ~numpy.ndarray): The value of padded pixels.
            If it is :class:`numpy.ndarray`,
            its shape should be :math:`(C, 1, 1)`,
            where :math:`C` is the number of channels of :obj:`img`.

    Returns:
        ~numpy.ndarray:
        An image array in CHW format.

    """
    B, C, H, W = imgs.shape
    n_col = min(n_col, B)
    n_row = int(math.ceil(B / n_col))

    shape = (C,
             (H + pad) * n_row,
             (W + pad) * n_col)
    grid = np.empty(shape, dtype=imgs.dtype)
    grid[:] = np.array(fill).reshape((-1, 1, 1))

    k = 0
    for y in range(n_row):
        for x in range(n_col):
            if k >= B:
                break
            start_y = y * (H + pad) + pad // 2 + 1
            start_x = x * (W + pad) + pad // 2 + 1
            grid[:,
                 start_y: start_y + H,
                 start_x: start_x + W] = imgs[k]
            k += 1

    return grid
