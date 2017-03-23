def center_crop(img, output_shape, return_slices=False, copy=False):
    """Center crop an image by `output_shape`.

    An image is cropped to :obj:`output_shape`. The center of the output image
    and the center of the input image are same.

    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        output_shape (tuple): the size of output image after cropping.
            This value is :math:`(heihgt, width)`.
        return_slices (bool): If :obj:`True`, this function returns information
            of slices.
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
    _, H, W = img.shape
    oH, oW = output_shape
    if oH > H or oW > W:
        raise ValueError('shape of image needs to be larger than output_shape')

    start_H = int(round((H - oH) / 2.))
    start_W = int(round((W - oW) / 2.))

    slice_H = slice(start_H, start_H + oH)
    slice_W = slice(start_W, start_W + oW)

    img = img[:, slice_H, slice_W]

    if copy:
        img = img.copy()

    if return_slices:
        return img, slice_H, slice_W
    else:
        return img
