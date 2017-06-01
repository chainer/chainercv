def flip_keypoint(keypoint, size, y_flip=False, x_flip=False):
    """Modify keypoints according to image flips.

    Args:
        keypoint (~numpy.ndarray): Keypoints in the image.
            The shape of this array is :math:`(K, 2)`. :math:`K` is the number
            of keypoints in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the keypoints.
        size (tuple): A tuple of length 2. The height and the width
            of the image which is associated with the keypoints.
        y_flip (bool): Modify keypoints according to a vertical flip of
            an image.
        x_flip (bool): Modify keypoints according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Keypoints modified according to image flips.

    """
    H, W = size
    keypoint = keypoint.copy()
    if y_flip:
        keypoint[:, 0] = H - 1 - keypoint[:, 0]
    if x_flip:
        keypoint[:, 1] = W - 1 - keypoint[:, 1]
    return keypoint
