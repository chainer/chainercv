def flip(img, x_flip=False, y_flip=False, copy=False):
    """Flip an image in vertical or horizontal direction as specified.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in CHW
            format.
        x_flip (bool): Flip in horizontal direction.
        y_flip (bool): Flip in vertical direction.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        Transformed :obj:`img` in CHW format.
    """
    if x_flip:
        img = img[:, :, ::-1]
    if y_flip:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()
    return img
