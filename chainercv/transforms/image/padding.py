from __future__ import division

import numpy as np


def padding(img, size, fill=0, return_param=False):
    """Add pixels around an image.

    This method place the input image at the center of a larger canvas.
    The size of the canvas is :obj:`size`. The canvas is filled by a value
    :obj:`fill` except for the region where the original image is placed.

    This transformation is commonly used a preprocess of random crop [#]_.

    .. [#] Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, \
    Zhuowen Tu. \
    Deeply-supervised nets. Artificial Intelligence and Statistics. 2015.

    Args:
        img (~numpy.ndarray): An image array to be padded. This is in
            CHW format.
        size (tuple): The size of output image after padding.
            This value is :math:`(width, height)`.
        fill (float, tuple or ~numpy.ndarray): The value of padded pixels.
            The default value is :obj:`0`.
        return_param (bool): If :obj:`True`, this function returns information
            of slices.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of padding.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **x_offset** (*int*): The x coordinate of the top left corner\
            of the image after placing on the canvas.
        * **y_offset** (*int*): The y coodinate of the top left corner of\
            the image after placing on the canvas.

    """

    C, H, W = img.shape
    out_W, out_H = size
    if out_W < W or out_H < H:
        raise ValueError('shape of image needs to be smaller than size')

    x_offset = (out_W - W) // 2
    y_offset = (out_H - H) // 2

    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape(-1, 1, 1)
    out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

    if return_param:
        param = {'x_offset': x_offset, 'y_offset': y_offset}
        return out_img, param
    else:
        return out_img
