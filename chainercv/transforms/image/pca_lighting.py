import numpy


def pca_lighting(img, sigma, eigen_value=None, eigen_vector=None):
    """Alter the intensities of input image using PCA.

    This is used in training of AlexNet [Krizhevsky]_.

    .. [Krizhevsky] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. \
    ImageNet Classification with Deep Convolutional Neural Networks. \
    NIPS 2012."ImageNet Classification with Deep Convolutional Neural
    Networks" [NIPS 2012].

    Args:
        image (numpy.ndarray): An image array to be augmented. This is in
            CHW format.
        sigma (float): Standard deviation of the Gaussian. In AlexNet
            [Krizhevsky]_, this value is 10% of the range of intensity
            (25.5 if the range is [0, 255]).
        eigen_value: (numpy.ndarray): An array of eigen values. The shape
            have to be (3,). If it is not specified, the values computed from
            ImageNet is used.
        eigen_vector: (numpy.ndarray): An array of eigen vectors. The shape
            have to be (3, 3). If it is not specified, the vectors computed
            from ImageNet is used.

    Returns:
        An image in CHW format.
    """

    if sigma <= 0:
        return img

    # these values are copied from facebook/fb.resnet.torch
    if eigen_value is None:
        eigen_value = numpy.array((0.2175, 0.0188, 0.0045))
    if eigen_vector is None:
        eigen_vector = numpy.array((
            (0.4009, -0.814,  0.4203),
            (0.7192, -0.0045, -0.6948),
            (-0.5675, -0.5808, -0.5836)))

    alpha = numpy.random.normal(0, sigma, size=3)

    img = img.copy()
    img += eigen_vector.dot(eigen_value * alpha).reshape(-1, 1, 1)

    return img
