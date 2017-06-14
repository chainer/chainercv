import numpy as np


def assert_is_bbox(bbox):
    """Checks if bounding boxes satisfy bounding box format.

    This function checks if given bounding boxes satisfy bounding boxe
    format or not.
    If the bounding boxes do not satifiy the format, this function raises an
    :class:`AssertionError`.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be checked.

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
