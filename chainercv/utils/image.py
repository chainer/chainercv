import numpy as np
from PIL import Image


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def write_image(img, path):
    """Save an image to a file.

    This function saves an image to given file. The image is in CHW format and
    the range of its value is :math:`[0, 255]`.

    Args:
        image (~numpy.ndarray): An image to be saved.
        path (str): The path of an image file.

    """

    if img.shape[0] == 1:
        img = img[0]
    else:
        img = img.transpose((1, 2, 0))

    img = Image.fromarray(img.astype(np.uint8))
    img.save(path)
