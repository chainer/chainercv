import random
import six


def random_crop(img, size, return_param=False, copy=False):
    """Crop array randomly into `size`.

    The input image is cropped by a randomly selected region whose shape
    is :obj:`size`.

    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        size (tuple): The size of output image after cropping.
            This value is :math:`(height, width)`.
        return_param (bool): If :obj:`True`, this function returns
            information of slices.
        copy (bool): If :obj:`False`, a view of :obj:`img` is returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is cropped from the input
        array.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_slice** (*slice*): A slice used to crop the input image.\
            The relation below holds together with :obj:`x_slice`.
        * **x_slice** (*slice*): Similar to :obj:`x_slice`.

            .. code::

                out_img = img[:, y_slice, x_slice]

    """
    H, W = size

    if img.shape[1] == H:
        y_offset = 0
    elif img.shape[1] > H:
        y_offset = random.choice(six.moves.range(img.shape[1] - H))
    else:
        raise ValueError('shape of image needs to be larger than output shape')
    y_slice = slice(y_offset, y_offset + H)

    if img.shape[2] == W:
        x_offset = 0
    elif img.shape[2] > W:
        x_offset = random.choice(six.moves.range(img.shape[2] - W))
    else:
        raise ValueError('shape of image needs to be larger than output shape')
    x_slice = slice(x_offset, x_offset + W)

    img = img[:, y_slice, x_slice]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_slice': y_slice, 'x_slice': x_slice}
    else:
        return img
