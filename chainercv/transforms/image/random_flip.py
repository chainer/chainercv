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
        This function returns :obj:`out_img, x_flip, y_flip` if
        :obj:`return_flip = True`. Otherwise, this returns :obj:`out_img`.

        Note that :obj:`out_img` is the transformed image array.
        Also, :obj:`x_flip` and :obj:`y_flip` are bools that indicate whether
        the image was flipped in the horizontal direction and the vertical
        direction respectively.

    """
    x_flip, y_flip = False, False
    if random_x:
        x_flip = random.choice([True, False])
    if random_y:
        y_flip = random.choice([True, False])

    if x_flip:
        img = img[:, :, ::-1]
    if y_flip:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()

    if return_flip:
        return img, x_flip, y_flip
    else:
        return img
