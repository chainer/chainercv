def flip_bbox(bbox, img_shape, h_flip=False, v_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :obj:`(x_min, y_min, x_max, y_max)`,
    where first four attributes are coordinates of the bottom left and the
    top right vertices. 

    Args:
        bbox (~numpy.ndarray): shape is :math:`(R, 4)`. :math:`R` is
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
    bbox = bbox.copy()
    if h_flip:
        x_max = W - 1 - bbox[:, 0]
        x_min = W - 1 - bbox[:, 2]
        bbox[:, 0] = x_min
        bbox[:, 2] = x_max
    if v_flip:
        y_max = H - 1 - bbox[:, 1]
        y_min = H - 1 - bbox[:, 3]
        bbox[:, 1] = y_min
        bbox[:, 3] = y_max
    return bbox
