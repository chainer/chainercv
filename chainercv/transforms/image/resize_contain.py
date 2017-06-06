import numpy as np

from chainercv.transforms import resize


def resize_contain(img, size, fill=0, return_param=False):
    """Resize the image to fit in the given area while keeping aspect ratio.

    If both the height and the width in :obj:`size` are larger than
    the height and the width of the :obj:`img`, the :obj:`img` is placed on
    the center with an appropriate padding to match :obj:`size`.

    Otherwise, the input image is scaled to fit in a canvas whose size
    is :obj:`size` while preserving aspect ratio.

    Args:
        img (~numpy.ndarray): An array to be transformed. This is in
            CHW format.
        size (tuple of two ints): A tuple of two elements:
            :obj:`height, width`. The size of the image after resizing.
        fill (float, tuple or ~numpy.ndarray): The value of padded pixels.
        return_param (bool): Returns information of resizing and offsetting.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of resizing.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_offset** (*int*): The y coodinate of the top left corner of\
            the image after placing on the canvas.
        * **x_offset** (*int*): The x coordinate of the top left corner\
            of the image after placing on the canvas.
        * **scaled_size** (*tuple*): The size to which the image is scaled\
            to before placing it on a canvas. This is a tuple of two elements:\
            :obj:`height, width`.

    """
    C, H, W = img.shape
    out_H, out_W = size
    scale_h = out_H / float(H)
    scale_w = out_W / float(W)
    scale = min(min(scale_h, scale_w), 1.)
    scaled_size = (int(H * scale), int(W * scale))
    if scale < 1.:
        img = resize(img, scaled_size)
    y_slice, x_slice = _get_pad_slice(img, size=size)
    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape(-1, 1, 1)
    out_img[:, y_slice, x_slice] = img

    if return_param:
        param = {'y_offset': y_slice.start, 'x_offset': x_slice.start,
                 'scaled_size': scaled_size}
        return out_img, param
    else:
        return out_img


def _get_pad_slice(img, size):
    """Get slices needed for padding.

    Args:
        img (~numpy.ndarray): This image is in format CHW.
        size (tuple of two ints): (max_H, max_W).
    """
    _, H, W = img.shape

    if H < size[0]:
        diff_y = size[0] - H
        margin_y = diff_y / 2
        if diff_y % 2 == 0:
            y_slice = slice(int(margin_y), int(size[0] - margin_y))
        else:
            y_slice = slice(int(margin_y), int(size[0] - margin_y - 1))
    else:
        y_slice = slice(0, int(size[0]))

    if W < size[1]:
        diff_x = size[1] - W
        margin_x = diff_x / 2
        if diff_x % 2 == 0:
            x_slice = slice(int(margin_x), int(size[1] - margin_x))
        else:
            x_slice = slice(int(margin_x), int(size[1] - margin_x - 1))
    else:
        x_slice = slice(0, int(size[1]))

    return y_slice, x_slice
