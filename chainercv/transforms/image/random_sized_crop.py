from __future__ import division

import math
import numpy as np
import random


def random_sized_crop(img,
                      scale_ratio_interval=(0.08, 1),
                      aspect_ratio_interval=(3 / 4, 4 / 3),
                      return_params=False, copy=False):
    """Crop an image to random size and aspect ratio.

    The size :math:`(H_{crop}, W_{crop})` and the left top coordinate
    :math:`(y_{start}, x_{start})` of the crop are calculated as follows:

    + :math:`H_{crop} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\times a}}\\rfloor`
    + :math:`W_{crop} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\div a}}\\rfloor`
    + :math:`y_{start} \\sim Uniform(0, H - H_{crop})`
    + :math:`x_{start} \\sim Uniform(0, W - W_{crop})`
    + :math:`s \\sim Uniform(s_1, s_2)`
    + :math:`b \\sim Uniform(a_1, a_2)` and \
        :math:`a = b` or :math:`a = \\frac{1}{b}` in 50/50 probability.

    Here, :math:`s_1, s_2` are the two floats in
    :obj:`scale_ratio_interval` and :math:`a_1, a_2` are the two floats
    in :obj:`aspect_ratio_interval`.
    Also, :math:`H` and :math:`W` are the height and the width of the image.

    Args:
        img (~numpy.ndarray): An image array. This is in CHW format.
        scale_ratio_interval (tuple of two floats): Determines
            the distribution from which a scale ratio is sampled.
            The default values are selected so that the area of the crop is
            8~100% of the original image. This is the default
            setting used to train ResNets in Torch style.
        aspect_ratio_interval (tuple of two floats): Determines
            the distribution from which an aspect ratios is sampled.
            The default values are
            :math:`\\frac{3}{4}` and :math:`\\frac{4}{3}`, which
            are also the default setting to train ResNets in Torch style.
        return_param (bool): Returns parameters if :obj:`True`.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns only the cropped image.

        If :obj:`return_param = True`,
        returns a tuple of cropped image and :obj:`param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_slice** (*slice*): A slice used to crop the input image.\
            The relation below holds together with :obj:`x_slice`.
        * **x_slice** (*slice*): Similar to :obj:`y_slice`.

            .. code::

                out_img = img[:, y_slice, x_slice]

        * **scale_ratio** (float): :math:`s` in the description (see above).
        * **aspect_ratio** (float): :math:`a` in the description.

    """
    _, H, W = img.shape
    scale_ratio, aspect_ratio =\
        _sample_parameters(
            (H, W), scale_ratio_interval, aspect_ratio_interval)

    H_crop = int(math.floor(np.sqrt(scale_ratio * H * W * aspect_ratio)))
    W_crop = int(math.floor(np.sqrt(scale_ratio * H * W / aspect_ratio)))
    y_start = random.randint(0, H - H_crop)
    x_start = random.randint(0, W - W_crop)
    y_slice = slice(y_start, y_start + H_crop)
    x_slice = slice(x_start, x_start + W_crop)

    img = img[:, y_slice, x_slice]

    if copy:
        img = img.copy()
    if return_params:
        params = {'y_slice': y_slice, 'x_slice': x_slice,
                  'scale_ratio': scale_ratio, 'aspect_ratio': aspect_ratio}
        return img, params
    else:
        return img


def _sample_parameters(size, scale_ratio_interval, aspect_ratio_interval):
    H, W = size
    for _ in range(10):
        aspect_ratio = random.uniform(
            aspect_ratio_interval[0], aspect_ratio_interval[1])
        if random.uniform(0, 1) < 0.5:
            aspect_ratio = 1 / aspect_ratio
        # This is determined so that relationships "H - H_crop >= 0" and
        # "W - W_crop >= 0" are always satisfied.
        scale_ratio_max = min((scale_ratio_interval[1],
                               H / (W * aspect_ratio),
                               (aspect_ratio * W) / H))

        scale_ratio = random.uniform(
            scale_ratio_interval[0], scale_ratio_interval[1])
        if scale_ratio <= scale_ratio_max:
            return scale_ratio, aspect_ratio

    # This is guaranteed to be a valid param.
    scale_ratio = random.uniform(
        scale_ratio_interval[0], scale_ratio_max)
    return scale_ratio, aspect_ratio
