def resize_point(point, in_size, out_size):
    """Adapt point coordinates to the rescaled image space.

    Args:
        point (~numpy.ndarray): Points in the image.
            The shape of this array is :math:`(P, 2)`. :math:`P` is the number
            of points in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the points.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Points rescaled according to the given image shapes.

    """
    point = point.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    point[:, 0] = y_scale * point[:, 0]
    point[:, 1] = x_scale * point[:, 1]
    return point
