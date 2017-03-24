import numpy as np


def pad(img, size, bg_value):
    """Pad image to match given size.

    Args:
        img (~numpy.ndarray): An array to be transformed. This is in
            CHW format.
        size (tuple of two ints): a tuple of two elements:
            :obj:`width, height`. The size of the image after padding.
        bg_value (scalar): value of the padded regions

    Returns:
        ~numpy.ndarray: a padded array in CHW format.

    """
    x_slices, y_slices = _get_pad_slices(img, size=size)
    out = bg_value * np.ones((img.shape[0],) + size, dtype=img.dtype)
    out[:, y_slices, x_slices] = img
    return out


def _get_pad_slices(img, size):
    """Get slices needed for padding.

    Args:
        img (~numpy.ndarray): this image is in format CHW.
        size (tuple of two ints): (max_W, max_H).
    """
    _, H, W = img.shape

    if W < size[0]:
        diff_x = size[0] - W
        margin_x = diff_x / 2
        if diff_x % 2 == 0:
            x_slices = slice(int(margin_x), int(size[0] - margin_x))
        else:
            x_slices = slice(int(margin_x), int(size[0] - margin_x - 1))
    else:
        x_slices = slice(0, int(size[0]))

    if H < size[1]:
        diff_y = size[1] - H
        margin_y = diff_y / 2
        if diff_y % 2 == 0:
            y_slices = slice(int(margin_y), int(size[1] - margin_y))
        else:
            y_slices = slice(int(margin_y), int(size[1] - margin_y - 1))
    else:
        y_slices = slice(0, int(size[1]))
    return x_slices, y_slices
