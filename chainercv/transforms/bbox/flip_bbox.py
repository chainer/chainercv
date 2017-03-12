def flip_bbox(bboxes, img_shape, h_flip=False, v_flip=False):
    """Flip bounding boxes accordingly.

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
        img_shape (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        h_flip (bool): Flip bounding box according to a horizontal flip of
            an image.
        v_flip (bool): Flip bounding box according to a vertical flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = img_shape
    bboxes = bboxes.copy()
    if h_flip:
        x_max = W - 1 - bboxes[:, 0]
        x_min = W - 1 - bboxes[:, 2]
        bboxes[:, 0] = x_min
        bboxes[:, 2] = x_max
    if v_flip:
        y_max = H - 1 - bboxes[:, 1]
        y_min = H - 1 - bboxes[:, 3]
        bboxes[:, 1] = y_min
        bboxes[:, 3] = y_max
    return bboxes
