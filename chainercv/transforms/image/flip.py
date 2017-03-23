def flip(img, flip_x=False, flip_y=False, copy=False):
    """Flip an image in vertical or horizontal direction as specified.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in CHW
            format.
        flip_x (bool): flip in horizontal direction
        flip_y (bool): flip in vertical direction
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        Transformed :obj:`img` in CHW format.
    """
    if flip_x:
        img = img[:, :, ::-1]
    if flip_y:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()
    return img
