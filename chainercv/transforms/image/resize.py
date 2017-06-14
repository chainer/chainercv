import numpy as np
import PIL
import warnings


try:
    import cv2

    def _resize(img, size, interpolation):
        img = img.transpose(1, 2, 0)
        if interpolation == PIL.Image.NEAREST:
            cv_interpolation = cv2.INTER_NEAREST
        elif interpolation == PIL.Image.BILINEAR:
            cv_interpolation = cv2.INTER_LINEAR
        elif interpolation == PIL.Image.BICUBIC:
            cv_interpolation = cv2.INTER_CUBIC
        elif interpolation == PIL.Image.LANCZOS:
            cv_interpolation = cv2.INTER_LANCZOS4
        H, W = size
        img = cv2.resize(img, dsize=(W, H), interpolation=cv_interpolation)

        # If input is a grayscale image, cv2 returns a two-dimentional array.
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        return img.transpose(2, 0, 1)

except ImportError:
    warnings.warn(
        'cv2 is not installed on your environment. '
        'ChainerCV will fall back on Pillow. '
        'Installation of cv2 is recommended for faster computation. ',
        RuntimeWarning)

    def _resize(img, size, interpolation):
        C = img.shape[0]
        H, W = size
        out = np.empty((C, H, W), dtype=img.dtype)
        for ch, out_ch in zip(img, out):
            ch = PIL.Image.fromarray(ch, mode='F')
            out_ch[:] = ch.resize((W, H), resample=interpolation)
        return out


def resize(img, size, interpolation=PIL.Image.BILINEAR):
    """Resize image to match the given shape.

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
        size (tuple): This is a tuple of length 2. Its elements are
            ordered as (height, width).
        interpolation (int): Determines sampling strategy. This is one of
            :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BILINEAR`,
            :obj:`PIL.Image.BICUBIC`, :obj:`PIL.Image.LANCZOS`.
            Bilinear interpolation is the default strategy.

    Returns:
        ~numpy.ndarray: A resize array in CHW format.

    """
    img = _resize(img, size, interpolation)
    return img
