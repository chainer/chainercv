import numpy as np


def assert_is_point(point, visible=None, size=None, n_point=None):
    """Checks if points satisfy the format.

    This function checks if given points satisfy the format and
    raises an :class:`AssertionError` when the points violate the convention.

    Args:
        point (~numpy.ndarray): Points to be checked.
        visible (~numpy.ndarray): Visibility of the points.
            If this is :obj:`None`, all points are regarded as visible.
        size (tuple of ints): The size of an image.
            If this argument is specified,
            the coordinates of visible points are checked to be within the image.
        n_point (int): If specified, the number of points in each object is
            expected to be :obj:`n_point`.
    """

    for i, pnt in enumerate(point):
        assert isinstance(pnt, np.ndarray), \
            'pnt must be a numpy.ndarray.'
        assert pnt.dtype == np.float32, \
            'The type of pnt must be numpy.float32.'
        assert pnt.shape[1:] == (2,), \
            'The shape of pnt must be (*, 2).'
        if n_point is not None:
            assert pnt.shape[0] == n_point, \
                'The number of points should always be n_point'

        if visible is not None:
            assert len(point) == len(visible), \
                'The length of point and visible should be the same.'
            vsble = visible[i]
            assert isinstance(vsble, np.ndarray), \
                'pnt should be a numpy.ndarray.'
            assert vsble.dtype == np.bool, \
                'The type of visible must be numpy.bool.'
            assert vsble.ndim == 1, \
                'The dimensionality of a visible must be one.'
            assert vsble.shape[0] == pnt.shape[0], \
                'The size of the first axis should be the same for ' \
                'corresponding pnt and vsble.'
            visible_pnt = pnt[vsble]
        else:
            visible_pnt = pnt

        if size is not None:
            assert (visible_pnt >= 0).all() and (visible_pnt <= size).all(),\
                'The coordinates of visible points ' \
                'should not exceed the size of image.'
