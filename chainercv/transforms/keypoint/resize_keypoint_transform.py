def resize_keypoint(keypoints, input_shape, output_shape):
    """Change values of keypoints according to paramters for resizing an image.

    The shape of keypoints is :math:`(K, 3)`. :math:`K` is the number of
    keypoints in the image.
    The last dimension is composed of :obj:`(x, y, valid)` in this order.
    These are discriptions of a corresponding keypoint.
    :obj;`x` and :obj:`y` are coordinates of the keypoint. :obj:`valid`
    is whether the keypoint is visible in the image or not.

    Args:
        keypoints (~numpy.ndarray): keypoints in the image. see description
            above.
        input_shape (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        output_shape (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Keypoints rescaled according to the given image shapes.

    """
    keypoints = keypoints.copy()
    h_scale = float(output_shape[1]) / input_shape[1]
    v_scale = float(output_shape[0]) / input_shape[0]
    keypoints[:, 0] = h_scale * keypoints[:, 0]
    keypoints[:, 1] = v_scale * keypoints[:, 1]
    return keypoints
