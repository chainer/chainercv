from chainercv.transforms import resize


def scale(img, size, fit='short'):
    """Rescales the input image to the given "size".

    When :obj:`fit == short`, the input image will be resized so that
    the shorter edge will be scaled to length :obj:`size` after
    resizing. For example, if the height of the image is larger than
    its width, image will be resized to (size * height / width, size).

    When :obj:`fit == long`, the input image will be resized so that
    the longer edge will be scaled to length :obj:`size` after
    resizing.

    Args:
        img (~numpy.ndarray): An image array to be scaled. This is in
            CHW format.
        size (int): The length of the smaller edge.
        fit ({'short', 'long'}): Determines whether to match the length
            of the shorter edge or the longer edge to :obj:`size`.

    Returns:
        ~numpy.ndarray: A scaled image in CHW format.

    """
    _, H, W = img.shape
    if (W <= H and W == size) or (H <= W and H == size):
        return img

    if fit == 'short':
        if W < H:
            out_size = (size, int(size * H / W))
        else:
            out_size = (int(size * W / H), size)
    elif fit == 'long':
        if W < H:
            out_size = (int(size * W / H), size)
        else:
            out_size = (size, int(size * H / W))
    else:
        raise ValueError('fit needs to be either \'short\' or \'long\'')
    return resize(img, out_size)
