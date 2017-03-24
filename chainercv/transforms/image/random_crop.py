import random
import six


def random_crop(img, output_shape, return_slices=False, copy=False):
    """Crop array randomly into `output_shape`.

    The input image is cropped by a randomly selected region whose shape
    is :obj:`output_shape`.

    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        output_shape (tuple): the size of output image after cropping.
            This value is :math:`(heihgt, width)`.
        return_slices (bool): If :obj:`True`, this function returns
            information of slices.
        copy (bool): If :obj:`False`, a view of :obj:`img` is returned.

    Returns:
        This function returns :obj:`out_img, x_slice, y_slice` if
        :obj:`return_slices = True`. Otherwise, this returns
        :obj:`out_img`.

        Note that :obj:`out_img` is the transformed image array.
        Also, :obj:`x_slice` and :obj:`y_slice` are slices used to crop the
        input image. The following relationship is satisfied.

        .. code::

            out_img = img[:, y_slice, x_slice]

    """
    W, H = output_shape

    if img.shape[2] == W:
        x_offset = 0
    elif img.shape[2] > W:
        x_offset = random.choice(six.moves.range(img.shape[2] - W))
    else:
        raise ValueError('shape of image needs to be larger than output shape')
    x_slice = slice(x_offset, x_offset + W)

    if img.shape[1] == H:
        y_offset = 0
    elif img.shape[1] > H:
        y_offset = random.choice(six.moves.range(img.shape[1] - H))
    else:
        raise ValueError('shape of image needs to be larger than output shape')
    y_slice = slice(y_offset, y_offset + H)

    img = img[:, y_slice, x_slice]

    if copy:
        img = img.copy()

    if return_slices:
        return img, x_slice, y_slice
    else:
        return img
