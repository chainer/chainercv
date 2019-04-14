import numpy as np


def resize_point(point, in_size, out_size):
    """Adapt point coordinates to the rescaled image space.

    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`point`, ":math:`(R, K, 2)` or :math:`[(K, 2)]`", \
        :obj:`float32`, ":math:`(y, x)`"

    Returns:
        ~numpy.ndarray or list of arrays:
        Points rescaled according to the given image shapes.

    """
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    if isinstance(point, np.ndarray):
        out_point = point.copy()
        out_point[:, :, 0] = y_scale * point[:, :, 0]
        out_point[:, :, 1] = x_scale * point[:, :, 1]
    else:
        out_point = []
        for pnt in point:
            out_pnt = pnt.copy()
            out_pnt[:, 0] = y_scale * pnt[:, 0]
            out_pnt[:, 1] = x_scale * pnt[:, 1]
            out_point.append(out_pnt)
    return out_point
