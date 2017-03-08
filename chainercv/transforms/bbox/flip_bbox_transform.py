def flip_bbox(bboxes, img_shape, h_flip=False, v_flip=False):
    """Flip bounding boxes accordingly.

    The boundig boxes are a
    collection of length 5 arrays. Each array contains values
    organized as (x_min, y_min, x_max, y_max, label_id).

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
