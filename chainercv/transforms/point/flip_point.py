import numpy as np


def flip_point(point, size, y_flip=False, x_flip=False):
    """Modify points according to image flips.

    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        size (tuple): A tuple of length 2. The height and the width
            of the image, which is associated with the points.
        y_flip (bool): Modify points according to a vertical flip of
            an image.
        x_flip (bool): Modify keypoipoints according to a horizontal flip of
            an image.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`point`, ":math:`[(K, 2)]` or :math:`(R, K, 2)`", \
        :obj:`float32`, ":math:`(y, x)`"

    Returns:
        ~numpy.ndarray or list of arrays:
        Points modified according to image flips.

    """
    H, W = size
    if isinstance(point, np.ndarray):
        out_point = point.copy()
        if y_flip:
            out_point[:, :, 0] = H - out_point[:, :, 0]
        if x_flip:
            out_point[:, :, 1] = W - out_point[:, :, 1]
    else:
        out_point = []
        for pnt in point:
            pnt = pnt.copy()
            if y_flip:
                pnt[:, 0] = H - pnt[:, 0]
            if x_flip:
                pnt[:, 1] = W - pnt[:, 1]
            out_point.append(pnt)
    return out_point
