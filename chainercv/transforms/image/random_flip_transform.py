import random


def random_flip(xs, horizontal_flip=False, vertical_flip=False,
                return_flip=False):
    """Randomly flip images in vertical or horizontal direction.

    Args:
        xs (tuple or list of arrays or an numpy.ndarray): Arrays that
            are flipped.
        horizontal_flip (bool): randomly flip in horizontal direction.
        vertical_flip (bool): randomly flip in vertical direction.
        return_flip (bool): returns information of flip.

    Returns:
        Transformed :obj:`xs` and information about flip.
        If :obj:`return_flip` is False, information about flip will not be
        returned. The information is a dictionary with key :obj:`h` and
        :obj:`v` whose values are boolean. The bools contain whether the
        images were flipped along the corresponding orientation.

    """
    force_array = False
    if not isinstance(xs, tuple):
        xs = (xs,)
        force_array = True

    h_flip, v_flip = False, False
    if horizontal_flip:
        h_flip = random.choice([True, False])
    if vertical_flip:
        v_flip = random.choice([True, False])

    outs = []
    for x in xs:
        if h_flip:
            x = x[:, :, ::-1]
        if v_flip:
            x = x[:, ::-1, :]
        outs.append(x)

    if force_array:
        outs = outs[0]
    else:
        outs = tuple(outs)

    if return_flip:
        return outs, {'h': h_flip, 'v': v_flip}
    else:
        return outs
