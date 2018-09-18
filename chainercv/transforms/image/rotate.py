import warnings


try:
    import scipy.ndimage
    _available = True
except ImportError:
    _available = False


def _check_available():
    if not _available:
        warnings.warn(
            'SciPy is not installed in your environment,'
            'so rotate cannot be loaded.'
            'Please install SciPy to load dataset.\n\n'
            '$ pip install scipy')


def rotate(img, angle, expand=True):
    """Rotate images by degrees.

    Args:
        img (~numpy.ndarray): An arrays that get rotated. This is in
            CHW format.
        angle (float): Counter clock-wise rotation angle (degree) in
            [-180, 180].
        expand (bool): The output shaped is adapted or not.
            If :obj:`True`, the input image is contained complete in
            the output.

    Returns:
        ~numpy.ndarray:
        returns an array :obj:`out_img` that is the result of rotation.

    """

    _check_available()

    assert 180 >= angle >= -180
    return scipy.ndimage.rotate(img, angle, axes=(2, 1), reshape=expand)
