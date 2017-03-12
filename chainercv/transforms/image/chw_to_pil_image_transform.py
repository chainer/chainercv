import numpy as np


def chw_to_pil_image(img, reverse_color_channel=True):
    """Transforms a CHW array into uint8 HWC array.

    Args:
        img (~numpy.ndarray): an array in CHW format.
        bgr_to_rgb (bool): If true, the array's color channel is
            reversed.

            .. code:: python

                img = img[:, :, ::-1]

    Returns:
        ~numpy.ndarray: a uint8 image array

    """
    img = img.transpose(1, 2, 0)
    if reverse_color_channel:
        img = img[:, :, ::-1]
    return img.astype(np.uint8)


def chw_to_pil_image_tuple(imgs, indices=[0], reverse_color_channel=True):
    """Transforms CHW arrays into uint8 HWC arrays.

    This function transforms selected arrays in a tuple into HWC arrays
    which are :obj:`dtype==numpy.uint8`.

    Args:
        imgs (tuple of numpy.ndarray): Tuple of numpy.ndarrays which are not
            limited to image arrays.
        indices (list of ints): The integers in :obj:`indices` point to
            arrays in :obj:`imgs` which will be converted.
        bgr_to_rgb (bool): If true, the array's color channel is
            reversed.

            .. code:: python

                img = img[:, :, ::-1]

    Returns:
        tuple of numpy.ndarray

    """
    imgs = list(imgs)
    for i in indices:
        imgs[i] = chw_to_pil_image(imgs[i], reverse_color_channel)
    return tuple(imgs)
