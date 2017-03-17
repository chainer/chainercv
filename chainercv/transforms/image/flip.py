def flip(img, horizontal=False, vertical=False, copy=False):
    """Flip an image in vertical or horizontal direction as specified.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in CHW format.
        horizontal (bool): flip in horizontal direction
        vertical (bool): flip in vertical direction
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        Transformed :obj:`img` in CHW format.
    """
    if horizontal:
        img = img[:, :, ::-1]
    if vertical:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()
    return img
