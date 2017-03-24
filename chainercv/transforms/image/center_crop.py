def center_crop(img, output_shape, return_slices=False, copy=False):
    """Center crop an image by `output_shape`.

    An image is cropped to :obj:`output_shape`. The center of the output image
    and the center of the input image are same.

    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        output_shape (tuple): the size of output image after cropping.
            This value is :math:`(width, height)`.
        return_slices (bool): If :obj:`True`, this function returns information
            of slices.
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
    _, H, W = img.shape
    oW, oH = output_shape
    if oW > W or oH > H:
        raise ValueError('shape of image needs to be larger than output_shape')

    x_offset = int(round((W - oW) / 2.))
    y_offset = int(round((H - oH) / 2.))

    x_slice = slice(x_offset, x_offset + oW)
    y_slice = slice(y_offset, y_offset + oH)

    img = img[:, y_slice, x_slice]

    if copy:
        img = img.copy()

    if return_slices:
        return img, x_slice, y_slice
    else:
        return img
