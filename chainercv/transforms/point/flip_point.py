def flip_point(point, size, y_flip=False, x_flip=False):
    """Modify points according to image flips.

    Args:
        point (~numpy.ndarray): Points in the image.
            The shape of this array is :math:`(P, 2)`. :math:`P` is the number
            of points in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the points.
        size (tuple): A tuple of length 2. The height and the width
            of the image, which is associated with the points.
        y_flip (bool): Modify points according to a vertical flip of
            an image.
        x_flip (bool): Modify keypoipoints according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Points modified according to image flips.

    """
    H, W = size
    point = point.copy()
    if y_flip:
        point[:, 0] = H - point[:, 0]
    if x_flip:
        point[:, 1] = W - point[:, 1]
    return point
