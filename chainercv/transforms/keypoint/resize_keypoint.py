def resize_keypoint(keypoint, in_size, out_size):
    """Change values of keypoint according to paramters for resizing an image.

    Args:
        keypoint (~numpy.ndarray): Keypoints in the image.
            The shape of this array is :math:`(K, 2)`. :math:`K` is the number
            of keypoint in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the keypoints.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Keypoint rescaled according to the given image shapes.

    """
    keypoint = keypoint.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    keypoint[:, 0] = y_scale * keypoint[:, 0]
    keypoint[:, 1] = x_scale * keypoint[:, 1]
    return keypoint
