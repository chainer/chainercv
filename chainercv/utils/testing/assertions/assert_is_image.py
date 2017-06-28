import numpy as np


def assert_is_image(img, color=True, check_range=True):
    """Checks if an image satisfies image format.

    This function checks if a given image satisfies image format or not.
    If the image does not satifiy the format, this function raises an
    :class:`AssertionError`.

    Args:
        img (~numpy.ndarray): An image to be checked.
        color (bool): A boolean that determines the expected channel size.
            If it is :obj:`True`, the number of channels
            should be :obj:`3`. Otherwise, it should be :obj:`1`.
            The default value is :obj:`True`.
        check_range (bool): A boolean that determines whether the range
            of values are checked or not. If it is :obj:`True`,
            The values of image must be in :math:`[0, 255]`.
            Otherwise, this function does not check the range.
            The default value is :obj:`True`.

    """

    assert isinstance(img, np.ndarray), 'img must be a numpy.ndarray.'
    assert len(img.shape) == 3, 'img must be a 3-dimensional array.'
    C, H, W = img.shape

    if color:
        assert C == 3, 'The number of channels must be 3.'
    else:
        assert C == 1, 'The number of channels must be 1.'

    if check_range:
        assert img.min() >= 0 and img.max() <= 255, \
            'The values of img must be in [0, 255].'
