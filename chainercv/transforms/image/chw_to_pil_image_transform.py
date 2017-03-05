import numpy as np


def chw_to_pil_image(x, reverse_color_channel=True):
    """Transforms a CHW array into uint8 HWC array.

    Args:
        x (~numpy.ndarray): an array in CHW format.
        bgr_to_rgb (bool): If true, the array's color channel is
            reversed.

            .. code:: python

                x = x[:, :, ::-1]

    Returns:
        ~numpy.ndarray:

    """
    x = x.transpose(1, 2, 0)
    if reverse_color_channel:
        x = x[:, :, ::-1]
    return x.astype(np.uint8)


def chw_to_pil_image_tuple(xs, indices=[0], reverse_color_channel=True):
    """Transforms CHW arrays into uint8 HWC arrays.

    This function transforms selected arrays in a tuple into HWC arrays
    which are :obj:`dtype==numpy.uint8`.

    Args:
        xs (tuple of numpy.ndarray): Tuple of numpy.ndarrays which are not
            limited to image arrays.
        indices (list of ints): The integers in :obj:`indices` point to
            arrays in :obj:`xs` which will be converted.
        bgr_to_rgb (bool): If true, the array's color channel is
            reversed.

            .. code:: python

                x = x[:, :, ::-1]

    Returns:
        tuple of numpy.ndarray

    """
    xs = list(xs)
    for i in indices:
        xs[i] = chw_to_pil_image(xs[i], reverse_color_channel)
    return tuple(xs)
