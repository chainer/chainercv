import warnings

try:
    import cv2

    def _resize(img, size):
        return cv2.resize(img, dsize=size)

except ImportError:
    import numpy
    import PIL

    warnings.warn(
        'cv2 is not installed on your environment. '
        'ChainerCV will fallback on Pillow. ',
        RuntimeWarning)

    def _resize(img, size):
        channels = []
        for i in range(3):
            ch = PIL.Image.fromarray(
                img[:, :, i], mode='F', resample=PIL.Image.BILINEAR)
            ch = ch.resize(size)
            channels.append(numpy.array(ch))
        return numpy.stack(channels, axis=2)


def resize(img, output_shape):
    """Resize image to match the given shape.

    A bilinear interpolation is used for resizing.

    This method uses `cv2` or `Pillow` for the backend.
    If `cv2` is installed, it uses `cv2`. This backend is highly
    recommended because it is faster than `Pillow`. If you are using Anaconda,
    you can install it by following command.
        $ conda install -c menpo opencv=2.4.11

    If `cv2` is not installed, this method uses `Pillow`.

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
