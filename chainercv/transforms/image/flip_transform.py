def flip(x, horizontal=False, vertical=False, copy=False):
    """Flip an image in vertical or horizontal direction as specified.

    Args:
        x (numpy.ndarray): an array that gets flipped.
        horizontal (bool): flip in horizontal direction
        vertical (bool): flip in vertical direction
        copy (bool): If False, a view of :obj:`x` will be returned.

    Returns:
        Transformed :obj:`x`.
    """
    if horizontal:
        x = x[:, :, ::-1]
    if vertical:
        x = x[:, ::-1, :]

    if copy:
        x = x.copy()
    return x
