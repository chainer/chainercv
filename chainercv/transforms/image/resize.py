import numpy as np
import warnings

try:
    import cv2

    def _resize(img, size):
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, dsize=size)

        # If input is a grayscale image, cv2 returns a two-dimentional array.
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        return img.transpose(2, 0, 1)

except ImportError:
    import PIL

    warnings.warn(
        'cv2 is not installed on your environment. '
        'ChainerCV will fall back on Pillow. '
        'Installation of cv2 is recommended for faster computation. ',
        RuntimeWarning)

    def _resize(img, size):
        C = img.shape[0]
        W, H = size
        out = np.empty((C, H, W), dtype=img.dtype)
        for ch, out_ch in zip(img, out):
            ch = PIL.Image.fromarray(ch, mode='F')
            out_ch[:] = ch.resize(size, resample=PIL.Image.BILINEAR)
        return out


def resize(img, output_shape):
    """Resize image to match the given shape.

    A bilinear interpolation is used for resizing.

    This method uses :mod:`cv2` or :mod:`PIL` for the backend.
    If :mod:`cv2` is installed, this function uses the implementation in
    :mod:`cv2`. This implementation is faster than the implementation in
    :mod:`PIL`. Under Anaconda environment,
    :mod:`cv2` can be installed by the following command.

    .. code::

        $ conda install -c menpo opencv3=3.2.0

    Args:
        img (~numpy.ndarray): An array to be transformed.
            This is in CHW format and the type should be :obj:`numpy.float32`.
        output_shape (tuple): this is a tuple of length 2. Its elements are
            ordered as (width, height).

    Returns:
        ~numpy.ndarray: A resize array in CHW format.

    """
    img = _resize(img, output_shape)
    return img
