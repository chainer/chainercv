import random


def random_flip(img, random_h=False, random_v=False,
                return_flip=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (numpy.ndarray): an array that gets flipped.
        random_h (bool): randomly flip in horizontal direction.
        random_v (bool): randomly flip in vertical direction.
        return_flip (bool): returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        Tuple of transformed :obj:`img` and information of flip if
        :obj:`return_flip = True`.
        If :obj:`return_flip = False`, information about flip will not be
        returned. The information is a dictionary with key :obj:`h` and
        :obj:`v` whose values are boolean. The bools contain whether the
        images were flipped along the corresponding orientation.

    """
    flip_h, flip_v = False, False
    if random_h:
        flip_h = random.choice([True, False])
    if random_v:
        flip_v = random.choice([True, False])

    if flip_h:
        img = img[:, :, ::-1]
    if flip_v:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()

    if return_flip:
        return img, {'h': flip_h, 'v': flip_v}
    else:
        return img
