import numpy as np
import random


def random_expand(img, max_ratio=4, fill=0, return_param=False):
    """Expand an image randomly.

    This method randomly place the input image on a larger canvas. The size of
    the canvas is :math:`(rH, rW)`, where :math:`(H, W)` is the size of the
    input image and :math:`r` is a random ratio drawn from
    :math:`[1, max\_ratio]`. The canvas is filled by a value :obj:`fill`
    except for the region where the original image is placed.

    This data augmentation trick is used to create "zoom out" effect [#]_.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, \
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg. \
    SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW format.
        max_ratio (float): The maximum ratio of expansion. In the original
            paper, this value is 4.
        fill (float, tuple or ~numpy.ndarray): The value of padded pixels.
            In the original paper, this value is the mean of ImageNet.
        return_param (bool): Returns random parameters.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of expansion.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **ratio** (*float*): The sampled value used to make the canvas.
        * **y_offset** (*int*): The y coodinate of the top left corner of\
            the image after placing on the canvas.
        * **x_offset** (*int*): The x coordinate of the top left corner\
            of the image after placing on the canvas.

    """

    if max_ratio <= 1:
        if return_param:
            return img, {'ratio': 1, 'y_offset': 0, 'x_offset': 0}
        else:
            return img

    C, H, W = img.shape

    ratio = random.uniform(1, max_ratio)
    out_H, out_W = int(H * ratio), int(W * ratio)

    y_offset = random.randint(0, out_H - H)
    x_offset = random.randint(0, out_W - W)

    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape(-1, 1, 1)
    out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

    if return_param:
        param = {'ratio': ratio, 'y_offset': y_offset, 'x_offset': x_offset}
        return out_img, param
    else:
        return out_img
