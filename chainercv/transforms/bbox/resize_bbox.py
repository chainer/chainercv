def resize_bbox(bbox, input_shape, output_shape):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :obj:`(x_min, y_min, x_max, y_max)`,
    where the four attributes are coordinates of the bottom left and the
    top right vertices.

    Args:
        bbox (~numpy.ndarray): shape is :math:`(R, 4)`. :math:`R` is
            the number of bounding boxes.
        input_shape (tuple): A tuple of length 2. The width and the height
            of the image before resized.
        output_shape (tuple): A tuple of length 2. The width and the height
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    x_scale = float(output_shape[0]) / input_shape[0]
    y_scale = float(output_shape[1]) / input_shape[1]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    return bbox
