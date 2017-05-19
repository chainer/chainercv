def resize_keypoint(keypoint, in_size, out_size):
    """Change values of keypoint according to paramters for resizing an image.

    Args:
        keypoint (~numpy.ndarray): Keypoints in the image.
            The shape of this array is :math:`(K, 2)`. :math:`K` is the number
            of keypoint in the image.
            The last dimension is composed of :math:`x` and :math:`y`
            coordinates of the keypoints.
        in_size (tuple): A tuple of length 2. The width and the height
            of the image before resized.
        out_size (tuple): A tuple of length 2. The width and the height
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Keypoint rescaled according to the given image shapes.

    """
    keypoint = keypoint.copy()
    x_scale = float(out_size[0]) / in_size[0]
    y_scale = float(out_size[1]) / in_size[1]
    keypoint[:, 0] = x_scale * keypoint[:, 0]
    keypoint[:, 1] = y_scale * keypoint[:, 1]
    return keypoint
