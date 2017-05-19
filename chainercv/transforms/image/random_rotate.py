import numpy as np


def random_rotate(img, return_param=False):
    """Randomly rotate images by 90, 180, 270 or 360 degrees.

    Args:
        img (~numpy.ndarray): An arrays that get flipped. This is in
            CHW format.
        return_param (bool): Returns information of rotation.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of rotation.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **k** (*int*): The integer that represents the number of\
            times the image is rotated by 90 degrees.

    """
    k = np.random.randint(4)
    img = np.transpose(img, axes=(1, 2, 0))
    img = np.rot90(img, k)
    img = np.transpose(img, axes=(2, 0, 1))
    if return_param:
        return img, {'k': k}
    else:
        return img
