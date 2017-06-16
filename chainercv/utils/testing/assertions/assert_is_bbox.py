import numpy as np


def assert_is_bbox(bbox, size=None):
    """Checks if bounding boxes satisfy bounding box format.

    This function checks if given bounding boxes satisfy bounding boxes
    format or not.
    If the bounding boxes do not satifiy the format, this function raises an
    :class:`AssertionError`.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be checked.
        size (tuple of ints): The size of an image.
            If this argument is specified,
            Each bounding box should be within the image.
    """

    assert isinstance(bbox, np.ndarray), \
        'bbox must be a numpy.ndarray.'
    assert bbox.dtype == np.float32, \
        'The type of bbox must be numpy.float32,'
    assert bbox.shape[1:] == (4,), \
        'The shape of bbox must be (*, 4).'
    assert (bbox[:, 0] < bbox[:, 2]).all(), \
        'The coordinate of top must be less than that of bottom.'
    assert (bbox[:, 1] < bbox[:, 3]).all(), \
        'The coordinate of left must be less than that of right.'

    if size is not None:
        assert (bbox[:, :2] >= 0).all() and (bbox[:, 2:] <= size).all(),\
            'The coordinates of bounding boxes ' \
            'should not exceed the size of image.'
