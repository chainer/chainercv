import numpy as np


def pad(x, max_size, bg_value):
    """Pad image to match given size.

    Args:
        x (~numpy.ndarray): array to be transformed
        max_size (tuple of two ints): the size of output image after
            padding (max_H, max_W).
        bg_value (scalar): value of the padded regions

    """
    x_slices, y_slices = _get_pad_slices(x, max_size=max_size)
    out = bg_value * np.ones((x.shape[0],) + max_size, dtype=x.dtype)
    out[:, y_slices, x_slices] = x
    return out


def _get_pad_slices(img, max_size):
    """Get slices needed for padding.

    Args:
        img (numpy.ndarray): this image is in format CHW.
        max_size (tuple of two ints): (max_H, max_W).
    """
    _, H, W = img.shape

    if H < max_size[0]:
        diff_y = max_size[0] - H
        margin_y = diff_y / 2
        if diff_y % 2 == 0:
            y_slices = slice(margin_y, max_size[0] - margin_y)
        else:
            y_slices = slice(margin_y, max_size[0] - margin_y - 1)
    else:
        y_slices = slice(0, max_size[0])

    if W < max_size[1]:
        diff_x = max_size[1] - W
        margin_x = diff_x / 2
        if diff_x % 2 == 0:
            x_slices = slice(margin_x, max_size[1] - margin_x)
        else:
            x_slices = slice(margin_x, max_size[1] - margin_x - 1)
    else:
        x_slices = slice(0, max_size[1])
    return x_slices, y_slices
