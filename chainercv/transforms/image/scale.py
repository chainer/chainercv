from chainercv.transforms import resize


def scale(img, size):
    """Rescales the input image to the given "size".

    If height of the image is larger than its width,
    image will be resized to (size * height / width, size).
    Similar resizing will be done otherwise.

    Args:
        img (~numpy.ndarray): An image array to be scaled. This is in
            CHW format.
        size (int): The length of the smaller edge.

    Returns:
        ~numpy.ndarray: A scaled image in CHW format.

    """
    _, H, W = img.shape
    if (W <= H and W == size) or (H <= W and H == size):
        return img

    if W < H:
        oH = int(size * H / W)
        return resize(img, (size, oH))
    else:
        oW = int(size * W / H)
        return resize(img, (oW, size))
