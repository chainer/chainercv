import numpy as np
import six


def _chw_to_pil_image(img, reverse_color_channel):
    img = img.transpose(1, 2, 0)
    if reverse_color_channel:
        img = img[:, :, ::-1]
    return img.astype(np.uint8)


def chw_to_pil_image(img, reverse_color_channel=True):
    """Transforms one or multiple CHW arrays into uint8 HWC arrays.

    This function transforms one or multiple CHW arrays into HWC format
    whose types :obj:`dtype==numpy.uint8`.

    Args:
        img (~numpy.ndarray, or tuple of arrays): An array or tuple of arrays
            which are in CHW format.
        bgr_to_rgb (bool): If true, the array's color channel is
            reversed.

            .. code:: python

                img = img[:, :, ::-1]

    Returns:
        ~numpy.ndarray or tuple of arrays: These arrays are in HWC\
            format and have uint8 as data type.

    """
    if not isinstance(img, tuple):
        return _chw_to_pil_image(img, reverse_color_channel)

    imgs = list(img)
    for i in six.moves.range(len(imgs)):
        imgs[i] = chw_to_pil_image(imgs[i], reverse_color_channel)
    return tuple(imgs)
