def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.

    Args:
        bbox (~numpy.ndarray): See the table below.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox
