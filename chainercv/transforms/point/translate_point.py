import numpy as np


def translate_point(point, y_offset=0, x_offset=0):
    """Translate points.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the top left point of the image
    to the coordinate :math:`(y, x) = (y_{offset}, x_{offset})`.

    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`point`, ":math:`[(K, 2)]` or :math:`(R, K, 2)`", \
        :obj:`float32`, ":math:`(y, x)`"

    Returns:
        ~numpy.ndarray:
        Points modified translation of an image.

    """

    if isinstance(point, np.ndarray):
        out_point = point.copy()

        out_point[:, :, 0] += y_offset
        out_point[:, :, 1] += x_offset
    else:
        out_point = []
        for pnt in point:
            out_pnt = pnt.copy()
            out_pnt[:, 0] += y_offset
            out_pnt[:, 1] += x_offset
            out_point.append(out_pnt)
    return out_point
