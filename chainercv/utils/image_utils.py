import numpy as np
from PIL import Image


def read_image(path, dtype=np.float32, copy=True, force_color=True):
    """Read image from file.

    This function reads image from given file. The image is CHW format. The
    range of value is :math:`[0, 255]`. If the image is color, the order of the
    channels is BGR.

    Args:
        path (str): Path of image file.
        dtype: The type of array. The default is :obj:`~numpy.float32`.
        copy (bool): Make the array mutable.
        force_color (bool): If :obj:`True`, the number of channels is 3.
            If :obj:`False`, it is same as that of input. The default is
            :obj:`True`.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if len(f.getbands()) == 1 and force_color:
            img = f.convert('RGB')
        else:
            img = f
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    if copy:
        # make the array editable
        img = img.copy()

    if img.ndim == 2:
        return img[np.newaxis]
    else:
        return img.transpose(2, 0, 1)[::-1]
