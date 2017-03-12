def resize_bbox(bboxes, input_shape, output_shape):
    """Resize bounding boxes according to image resize.

    The bounding box is expected to be a two dimensional tensor of shape
    :math:`(R, 5)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :obj:`(x_min, y_min, x_max, y_max, label_id)`, where first
    four attributes are coordinates of the bottom left and the top right
    vertices. The last attribute is the label id, which points to the
    category of the object in the bounding box.

    Args:
        bboxes (~numpy.ndarray): shape is :math:`(R, 5)`. :math:`R` is
            the number of bounding boxes.
        input_shape (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        output_shape (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bboxes = bboxes.copy()
    h_scale = float(output_shape[1]) / input_shape[1]
    v_scale = float(output_shape[0]) / input_shape[0]
    bboxes[:, 0] = h_scale * bboxes[:, 0]
    bboxes[:, 2] = h_scale * bboxes[:, 2]
    bboxes[:, 1] = v_scale * bboxes[:, 1]
    bboxes[:, 3] = v_scale * bboxes[:, 3]
    return bboxes
