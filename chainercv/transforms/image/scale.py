from chainercv.transforms import resize


def scale(img, size, fit_short=True):
    """Rescales the input image to the given "size".

    When :obj:`fit_short == True`, the input image will be resized so that
    the shorter edge will be scaled to length :obj:`size` after
    resizing. For example, if the height of the image is larger than
    its width, image will be resized to (size * height / width, size).

    Otherwise, the input image will be resized so that
    the longer edge will be scaled to length :obj:`size` after
    resizing.

    Args:
        img (~numpy.ndarray): An image array to be scaled. This is in
            CHW format.
        size (int): The length of the smaller edge.
        fit_short (bool): Determines whether to match the length
            of the shorter edge or the longer edge to :obj:`size`.

    Returns:
        ~numpy.ndarray: A scaled image in CHW format.

    """
    _, H, W = img.shape

    # If resizing is not necessary, return the input as is.
    if fit_short and (W <= H and W == size) or (H <= W and H == size):
        return img
    if not fit_short and (W >= H and W == size) or (H >= W and H == size):
        return img

    if fit_short:
        if W < H:
            out_size = (size, int(size * H / W))
        else:
            out_size = (int(size * W / H), size)
    else:
        if W < H:
            out_size = (int(size * W / H), size)
        else:
            out_size = (size, int(size * H / W))
    return resize(img, out_size)
