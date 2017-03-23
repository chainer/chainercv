import random


def random_flip(img, random_x=False, random_y=False,
                return_flip=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        random_x (bool): randomly flip in horizontal direction.
        random_y (bool): randomly flip in vertical direction.
        return_flip (bool): returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        This function returns :obj:`out_img, flip_x, flip_y` if
        :obj:`return_flip = True`. Otherwise, this returns :obj:`out_img`.

        Note that :obj:`out_img` is the transformed image array.
        Also, :obj:`flip_x` and :obj:`flip_y` are bools that indicate whether
        the image was flipped in the horizontal direction and the vertical
        direction respectively.

    """
    flip_x, flip_y = False, False
    if random_x:
        flip_x = random.choice([True, False])
    if random_y:
        flip_y = random.choice([True, False])

    if flip_x:
        img = img[:, :, ::-1]
    if flip_y:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()

    if return_flip:
        return img, flip_x, flip_y
    else:
        return img
