def translate_point(point, y_offset=0, x_offset=0):
    """Translate points.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the top left point of the image
    to the coordinate :math:`(y, x) = (y_{offset}, x_{offset})`.

    Args:
        point (~numpy.ndarray): Points in the image.
            The shape of this array is :math:`(P, 2)`. :math:`P` is the number
            of points in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the points.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Points modified translation of an image.

    """

    out_point = point.copy()

    out_point[:, 0] += y_offset
    out_point[:, 1] += x_offset

    return out_point
