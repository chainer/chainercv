def flip_keypoint(keypoint, size, x_flip=False, y_flip=False):
    """Modify keypoints according to image flips.

    Args:
        keypoint (~numpy.ndarray): Keypoints in the image.
            The shape of this array is :math:`(K, 2)`. :math:`K` is the number
            of keypoints in the image.
            The last dimension is composed of :math:`x` and :math:`y`
            coordinates of the keypoints.
        size (tuple): A tuple of length 2. The width and the height
            of the image which is associated with the keypoints.
        x_flip (bool): Modify keypoints according to a horizontal flip of
            an image.
        y_flip (bool): Modify keypoints according to a vertical flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Keypoints modified according to image flips.

    """
    W, H = size
    keypoint = keypoint.copy()
    if x_flip:
        keypoint[:, 0] = W - 1 - keypoint[:, 0]
    if y_flip:
        keypoint[:, 1] = H - 1 - keypoint[:, 1]
    return keypoint
