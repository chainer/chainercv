def translate_keypoint(keypoint, x_offset=0, y_offset=0):
    """Translate keypoints.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate :math:`(x\_offset, y\_offset)`.


    Args:
        keypoint (~numpy.ndarray): Keypoints in the image.
            The shape of this array is :math:`(K, 2)`. :math:`K` is the number
            of keypoints in the image.
            The last dimension is composed of :math:`x` and :math:`y`
            coordinates of the keypoints.
        x_offset (int or float): The offset along x axis.
        y_offset (int or float): The offset along y axis.

    Returns:
        ~numpy.ndarray:
        Keypoints modified translation of an image.

    """

    out_keypoint = keypoint.copy()

    out_keypoint[:, 0] += x_offset
    out_keypoint[:, 1] += y_offset

    return out_keypoint
