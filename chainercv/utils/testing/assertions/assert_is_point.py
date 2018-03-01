import numpy as np


def assert_is_point(point, mask=None, size=None):
    """Checks if points satisfy the format.

    This function checks if given points satisfy the format and
    raises an :class:`AssertionError` when the points violate the convention.

    Args:
        point (~numpy.ndarray): Points to be checked.
        mask (~numpy.ndarray): A mask of the points.
            If this is :obj:`None`, all points are regarded as valid.
        size (tuple of ints): The size of an image.
            If this argument is specified,
            the coordinates of valid points are checked to be within the image.
    """

    assert isinstance(point, np.ndarray), \
        'point must be a numpy.ndarray.'
    assert point.dtype == np.float32, \
        'The type of point must be numpy.float32.'
    assert point.shape[1:] == (2,), \
        'The shape of point must be (*, 2).'

    if mask is not None:
        assert isinstance(mask, np.ndarray), \
            'a mask of points must be a numpy.ndarray.'
        assert mask.dtype == np.bool, \
            'The type of mask must be numpy.bool.'
        assert mask.ndim == 1, \
            'The dimensionality of a mask must be one.'
        assert mask.shape[0] == point.shape[0], \
            'The size of the first axis should be the same for ' \
            'corresponding point and mask.'
        valid_point = point[mask]
    else:
        valid_point = point

    if size is not None:
        assert (valid_point >= 0).all() and (valid_point <= size).all(),\
            'The coordinates of valid points ' \
            'should not exceed the size of image.'
