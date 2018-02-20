from __future__ import division

import math
import numpy as np


def tile_images(imgs, n_col, pad=2, fill=0):
    """Make a tile of images

    Args:
        imgs (numpy.ndarray): A batch of images whose shape is BCHW.
        n_col (int): The number of columns in a tile.
        pad (int or tuple of two ints): :obj:`pad_y, pad_x`. This is the
            amounts of padding in y and x directions. If this is an integer,
            the amounts of padding in the two directions are the same.
            The default value is 2.
        fill (float, tuple or ~numpy.ndarray): The value of padded pixels.
            If it is :class:`numpy.ndarray`,
            its shape should be :math:`(C, 1, 1)`,
            where :math:`C` is the number of channels of :obj:`img`.

    Returns:
        ~numpy.ndarray:
        An image array in CHW format.
        The size of this image is
        :math:`((H + pad_{y}) \\times \\lceil B / n_{n_{col}} \\rceil,
        (W + pad_{x}) \\times n_{col})`.

    """
    if isinstance(pad, int):
        pad = (pad, pad)
    pad_y, pad_x = pad

    B, C, H, W = imgs.shape
    n_col = min(n_col, B)
    n_row = int(math.ceil(B / n_col))

    shape = (C,
             (H + pad_y) * n_row,
             (W + pad_x) * n_col)
    tile = np.empty(shape, dtype=imgs.dtype)
    tile[:] = np.array(fill).reshape((-1, 1, 1))

    k = 0
    for y in range(n_row):
        for x in range(n_col):
            if k >= B:
                break
            start_y = y * (H + pad_y) + pad_y // 2
            start_x = x * (W + pad_x) + pad_x // 2
            tile[:,
                 start_y: start_y + H,
                 start_x: start_x + W] = imgs[k]
            k += 1

    return tile
