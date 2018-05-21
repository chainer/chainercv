import numpy as np
from PIL import Image


def write_image(img, path):
    """Save an image to a file.

    This function saves an image to given file. The image is in CHW format and
    the range of its value is :math:`[0, 255]`.

    Args:
        image (~numpy.ndarray): An image to be saved.
        path (string): The path of an image file.

    """

    if img.shape[0] == 1:
        img = img[0]
    else:
        img = img.transpose((1, 2, 0))

    img = Image.fromarray(img.astype(np.uint8))
    img.save(path)
