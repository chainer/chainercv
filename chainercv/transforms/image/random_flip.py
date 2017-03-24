import random


def random_flip(img, x_random=False, y_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        x_random (bool): randomly flip in horizontal direction.
        y_random (bool): randomly flip in vertical direction.
        return_param (bool): returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        numpy.ndarray or tuple of numpy.ndarray and dict.

            If :obj:`return_param = False`,\
            returns an array :obj:`out_img` that is the result of flipping.

            If :obj:`return_param = True`,\
            returns a tuple whose elements are :obj:`out_img, param`.\
            :obj:`param` is a dictionary of intermediate parameters whose
            key and value-type pairs are as follows.

            * ('x_flip', *bool*): Whether the image was flipped in the
                horizontal direction or not.
            * ('y_flip', *bool*): Whether the image was flipped in the
                vertical direction or not.

    """
    x_flip, y_flip = False, False
    if x_random:
        x_flip = random.choice([True, False])
    if y_random:
        y_flip = random.choice([True, False])

    if x_flip:
        img = img[:, :, ::-1]
    if y_flip:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'x_flip': x_flip, 'y_flip': y_flip}
    else:
        return img
