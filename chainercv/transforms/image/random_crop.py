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
        This function returns :obj:`out_img, slice_H, slice_W` if
        :obj:`return_slices = True`. Otherwise, this returns
        :obj:`out_img`.

        Note that :obj:`out_img` is the transformed image array.
        Also, :obj:`slice_H` and :obj:`slice_W` are slices used to crop the
        input image. The following relationship is satisfied.

        .. code::

            out_img = img[:, slice_H, slice_W]

    """
    H, W = output_shape

    if img.shape[1] == H:
        start_H = 0
    elif img.shape[1] > H:
        start_H = random.choice(six.moves.range(img.shape[1] - H))
    else:
        raise ValueError('shape of image is larger than output shape')
    slice_H = slice(start_H, start_H + H)

    if img.shape[2] == W:
        start_W = 0
    elif img.shape[2] > W:
        start_W = random.choice(six.moves.range(img.shape[2] - W))
    else:
        raise ValueError('shape of image is larger than output shape')
    slice_W = slice(start_W, start_W + W)

    img = img[:, slice_H, slice_W]

    if copy:
        img = img.copy()

    if return_slices:
        return img, slice_H, slice_W
    else:
        return img
