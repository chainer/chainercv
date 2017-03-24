def resize_keypoint(keypoint, input_shape, output_shape):
    """Change values of keypoint according to paramters for resizing an image.

    Args:
        keypoint (~numpy.ndarray): Keypoints in the image.
            The shape of this array is :math:`(K, 2)`. :math:`K` is the number
            of keypoint in the image.
            The last dimension is composed of :math:`x` and :math:`y`
            coordinates of the keypoints.
        input_shape (tuple): A tuple of length 2. The width and the height
            of the image before resized.
        output_shape (tuple): A tuple of length 2. The width and the height
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        keypoint rescaled according to the given image shapes.

    """
    keypoint = keypoint.copy()
    h_scale = float(output_shape[0]) / input_shape[0]
    v_scale = float(output_shape[1]) / input_shape[1]
    keypoint[:, 0] = h_scale * keypoint[:, 0]
    keypoint[:, 1] = v_scale * keypoint[:, 1]
    return keypoint
