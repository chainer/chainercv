import numpy as np


def rotate_bbox(bbox, angle, size):
    """Rotate bounding boxes by degrees.

    Args:
        bbox (~numpy.ndarray): See the table below.
        angle (float): Counter clock-wise rotation angle (degree).
            image is rotated by 90 degrees.
        size (tuple): A tuple of length 2. The height and the width
            of the image.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given :obj:`k`.

    """
    if angle % 90 != 0:
        raise ValueError(
            'angle which satisfies angle % 90 == 0 is only supported: {}'
            .format(angle))
    H, W = size
    if angle % 360 == 0:
        return bbox

    if angle % 360 == 90:
        rotated_bbox = np.concatenate(
            (W - bbox[:, 3:4], bbox[:, 0:1],
             W - bbox[:, 1:2], bbox[:, 2:3]), axis=1)
    elif angle % 360 == 180:
        rotated_bbox = np.concatenate(
            (H - bbox[:, 2:3], W - bbox[:, 3:4],
             H - bbox[:, 0:1], W - bbox[:, 1:2]), axis=1)
    elif angle % 360 == 270:
        rotated_bbox = np.concatenate(
            (bbox[:, 1:2], H - bbox[:, 2:3],
             bbox[:, 3:4], H - bbox[:, 0:1]), axis=1)
    rotated_bbox = rotated_bbox.astype(bbox.dtype)
    return rotated_bbox
