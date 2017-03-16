import random


def random_flip(img, random_h=False, random_v=False,
                return_flip=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        random_h (bool): randomly flip in horizontal direction.
        random_v (bool): randomly flip in vertical direction.
        return_flip (bool): returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        This function returns :obj:`out_img, flip_h, flip_v` if
        :obj:`return_flip = True`. Otherwise, this returns :obj:`out_img`.

        Note that :obj:`out_img` is the transformed image array.
        Also, :obj:`flip_h` and :obj:`flip_v` are bools that indicate whether
        the image was flipped in the horizontal direction and the vertical
        direction respectively.

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
        return img, flip_h, flip_v
    else:
        return img
