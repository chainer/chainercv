import numpy
import random


def random_expand(img, max_ratio=4, fill=0, return_params=False):
    """Expand image randomly.

    This method expands the size of image randomly by padding pixels. The
    aspect ratio of the image is kept.

    This method is used in training of SSD [1].

    .. [1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, \
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg. \
    SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW format.
        max_ratio (float): The maximum ratio of expansion. In the original
            paper, this value is 4.
        fill (float or tuple or ~numpy.ndarray): The value of padded pixels.
            In the original paper, this value is the mean of ImageNet.
        return_params (bool): returns random parameters.

    Returns:
        This function returns :obj:`out_img, ratio, x_offset, y_offset` if
        :obj:`return_params=True`. Otherwise, this returns :obj:`out_img`.

    """

    if max_ratio <= 1:
        if return_params:
            # img, ratio, x_offset, y_offset
            return img, 1, 0, 0
        else:
            return img

    C, H, W = img.shape

    ratio = random.uniform(1, max_ratio)
    out_H, out_W = int(H * ratio), int(W * ratio)

    x_offset = random.randint(0, out_W - W)
    y_offset = random.randint(0, out_H - H)

    out_img = numpy.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = numpy.array(fill).reshape(-1, 1, 1)
    out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

    if return_params:
        return out_img, ratio, x_offset, y_offset
    else:
        return out_img
