try:
    import cv2
    _available = True

except ImportError:
    _available = False


def resize(x, output_shape):
    """Resize image to match the given shape.

    A bilinear interpolation is used for resizing.

    Args:
        x (~numpy.ndarray): array to be transformed. This is in CHW format.
        output_shape (tuple): this is a tuple of length 2. Its elements are
            ordered as (height, width).

    Returns:
        ~numpy.ndarray: a resize array

    """
    if not _available:
        raise ValueError('cv2 is not installed on your environment, '
                         'so nothing will be plotted at this time. '
                         'Please install OpenCV.\n\n Under Anaconda '
                         ' environment, you can install it by '
                         '$ conda install -c menpo opencv=2.4.11\n')

    H, W = output_shape
    x = x.transpose(1, 2, 0)
    x = cv2.resize(x, dsize=(W, H))
    x = x.transpose(2, 0, 1)
    return x
