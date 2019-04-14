import numpy as np
from PIL import Image


def write_image(img, file, format=None):
    """Save an image to a file.

    This function saves an image to given file. The image is in CHW format and
    the range of its value is :math:`[0, 255]`.

    Args:
        image (~numpy.ndarray): An image to be saved.
        file (string or file-like object): A path of image file or
            a file-like object of image.
        format (:obj:`{'bmp', 'jpeg', 'png'}`): The format of image.
            If :obj:`file` is a file-like object,
            this option must be specified.
    """

    if img.shape[0] == 1:
        img = img[0]
    else:
        img = img.transpose((1, 2, 0))

    img = Image.fromarray(img.astype(np.uint8))
    img.save(file, format=format)
