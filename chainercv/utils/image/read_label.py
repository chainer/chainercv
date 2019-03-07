import numpy as np
from PIL import Image


def read_label(file, dtype=np.int32):
    """Read a label image from a file.

    This function reads an label image from given file. If reading label
    doesn't work collectly, try :func:`~chainercv.utils.read_image` with
    a parameter :obj:`color=True`.

    Args:
        file (string or file-like object): A path of image file or
            a file-like object of image.
        dtype: The type of array. The default value is :obj:`~numpy.int32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(file)
    try:
        img = f.convert('P')
        img = np.array(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        return img
    elif img.shape[2] == 1:
        return img[:, :, 0]
    else:
        raise ValueError("Color image can't be accepted as label image.")
