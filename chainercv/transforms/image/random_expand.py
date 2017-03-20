import numpy as np
import random


def random_expand(img, max_ratio=4, fill=0, return_params=False):
    """Expand an image randomly.

    This method randomly place the input image on a larger canvas. The size of
    the canvas is :math:`(rW, rH)`, where :math:`(W, H)` is the size of the
    input image and :math:`r` is a random ratio drawn from
    :math:`[1, max\_ratio]`. The canvas is filled by a value :obj:`fill`
    except for the region where the original image is placed.

    This data augmentation trick is used to create "zoom out" effect [1].

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

    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape(-1, 1, 1)
    out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

    if return_params:
        return out_img, ratio, x_offset, y_offset
    else:
        return out_img
