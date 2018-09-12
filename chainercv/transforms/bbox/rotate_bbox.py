import numpy as np


def rotate_bbox(bbox, k, size):
    """Rotate bounding boxes by 90, 180, 270 or 360 degrees.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        k (int): The integer that represents the number of times the
            image is rotated by 90 degrees.
        size (tuple): A tuple of length 2. The height and the width
            of the image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given :obj:`k`.

    """
    if k % 4 == 0:
        return bbox
    H, W = size
    if k % 4 == 1:
        rotated_bbox = np.concatenate(
            (W - bbox[:, 3:4], bbox[:, 0:1],
             W - bbox[:, 1:2], bbox[:, 2:3]), axis=1)
    elif k % 4 == 2:
        rotated_bbox = np.concatenate(
            (H - bbox[:, 2:3], W - bbox[:, 3:4],
             H - bbox[:, 0:1], W - bbox[:, 1:2]), axis=1)
    elif k % 4 == 3:
        rotated_bbox = np.concatenate(
            (bbox[:, 1:2], H - bbox[:, 2:3],
             bbox[:, 3:4], H - bbox[:, 0:1]), axis=1)
    rotated_bbox = rotated_bbox.astype(bbox.dtype)
    return rotated_bbox
