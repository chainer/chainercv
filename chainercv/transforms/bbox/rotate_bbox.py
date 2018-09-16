import numpy as np


def rotate_bbox(bbox, angle, size):
    """Rotate bounding boxes by degrees.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        angle (float): Clock-wise rotation angle (degree) in [-180, 180].
            image is rotated by 90 degrees.
        size (tuple): A tuple of length 2. The height and the width
            of the image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given :obj:`k`.

    """
    if angle % 90 != 0:
        raise ValueError(
            'angle in [180, 90, 0, -90, -180] is only supported: {}'
            .format(angle))
        return bbox
    H, W = size
    if angle == 0:
        return bbox

    if angle == 90:
        rotated_bbox = np.concatenate(
            (W - bbox[:, 3:4], bbox[:, 0:1],
             W - bbox[:, 1:2], bbox[:, 2:3]), axis=1)
    elif angle in [180, -180]:
        rotated_bbox = np.concatenate(
            (H - bbox[:, 2:3], W - bbox[:, 3:4],
             H - bbox[:, 0:1], W - bbox[:, 1:2]), axis=1)
    elif angle == -90:
        rotated_bbox = np.concatenate(
            (bbox[:, 1:2], H - bbox[:, 2:3],
             bbox[:, 3:4], H - bbox[:, 0:1]), axis=1)
    rotated_bbox = rotated_bbox.astype(bbox.dtype)
    return rotated_bbox