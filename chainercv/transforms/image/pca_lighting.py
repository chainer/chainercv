import numpy as np


def pca_lighting(img, sigma, eigen_value=None, eigen_vector=None):
    """AlexNet style color augmentation

    This method adds a noise vector drawn from a Gaussian. The direction of
    the Gaussian is same as that of the principal components of the dataset.

    This method is used in training of AlexNet [#]_.

    .. [#] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. \
    ImageNet Classification with Deep Convolutional Neural Networks. \
    NIPS 2012.

    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW and RGB format.
        sigma (float): Standard deviation of the Gaussian. In the original
            paper, this value is 10% of the range of intensity
            (25.5 if the range is :math:`[0, 255]`).
        eigen_value (~numpy.ndarray): An array of eigen values. The shape
            has to be :math:`(3,)`. If it is not specified, the values computed
            from ImageNet are used.
        eigen_vector (~numpy.ndarray): An array of eigen vectors. The shape
            has to be :math:`(3, 3)`. If it is not specified, the vectors
            computed from ImageNet are used.

    Returns:
        An image in CHW format.
    """

    if sigma <= 0:
        return img

    # these values are copied from facebook/fb.resnet.torch
    if eigen_value is None:
        eigen_value = np.array((0.2175, 0.0188, 0.0045))
    if eigen_vector is None:
        eigen_vector = np.array((
            (-0.5675, -0.5808, -0.5836),
            (0.7192, -0.0045, -0.6948),
            (0.4009, -0.814,  0.4203)))

    alpha = np.random.normal(0, sigma, size=3)

    img = img.copy()
    img += eigen_vector.dot(eigen_value * alpha).reshape(-1, 1, 1)

    return img
