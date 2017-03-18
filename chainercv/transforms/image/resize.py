import numpy
import warnings

try:
    import cv2

    def _resize(img, size):
        img = cv2.resize(img, dsize=size)
        if len(img.shape) == 3:
            return img
        else:
            return img[:, :, numpy.newaxis]

except ImportError:
    import PIL

    warnings.warn(
        'cv2 is not installed on your environment. '
        'ChainerCV will fall back on Pillow. '
        'Installation of cv2 is recommended for faster computation. ',
        RuntimeWarning)

    def _resize(img, size):
        channels = []
        for i in range(img.shape[2]):
            ch = PIL.Image.fromarray(img[:, :, i], mode='F')
            ch = ch.resize(size, resample=PIL.Image.BILINEAR)
            channels.append(numpy.array(ch))
        return numpy.stack(channels, axis=2)


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
            ordered as (height, width).

    Returns:
        ~numpy.ndarray: A resize array in CHW format.

    """
    H, W = output_shape
    img = img.transpose(1, 2, 0)
    img = _resize(img, (W, H))
    img = img.transpose(2, 0, 1)
    return img
