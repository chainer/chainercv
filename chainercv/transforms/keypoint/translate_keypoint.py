def translate_keypoint(keypoint, y_offset=0, x_offset=0):
    """Translate keypoints.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the top left point of the image
    to the coordinate :math:`(y, x) = (y\_offset, x\_offset)`.

    Args:
        keypoint (~numpy.ndarray): Keypoints in the image.
            The shape of this array is :math:`(K, 2)`. :math:`K` is the number
            of keypoints in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the keypoints.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Keypoints modified translation of an image.

    """

    out_keypoint = keypoint.copy()

    out_keypoint[:, 0] += y_offset
    out_keypoint[:, 1] += x_offset

    return out_keypoint
