import PIL
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


def rotate(img, angle, expand=True, interpolation=PIL.Image.BILINEAR):
    """Rotate images by degrees.

    Args:
        img (~numpy.ndarray): An arrays that get rotated. This is in
            CHW format.
        angle (float): Counter clock-wise rotation angle (degree).
        expand (bool): The output shaped is adapted or not.
            If :obj:`True`, the input image is contained complete in
            the output.
        interpolation (int): Determines sampling strategy. This is one of
            :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BILINEAR`,
            :obj:`PIL.Image.BICUBIC`.
            Bilinear interpolation is the default strategy.

    Returns:
        ~numpy.ndarray:
        returns an array :obj:`out_img` that is the result of rotation.

    """

    _check_available()

    # http://scikit-image.org/docs/dev/api/skimage.transform.html#warp
    if interpolation == PIL.Image.NEAREST:
        interpolation_order = 0
    elif interpolation == PIL.Image.BILINEAR:
        interpolation_order = 1
    elif interpolation == PIL.Image.BICUBIC:
        interpolation_order = 3

    return scipy.ndimage.rotate(
        img, angle, axes=(2, 1), reshape=expand,
        order=interpolation_order)
