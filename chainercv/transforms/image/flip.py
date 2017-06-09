def flip(img, y_flip=False, x_flip=False, copy=False):
    """Flip an image in vertical or horizontal direction as specified.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in CHW
            format.
        y_flip (bool): Flip in vertical direction.
        x_flip (bool): Flip in horizontal direction.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        Transformed :obj:`img` in CHW format.
    """
    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()
    return img
