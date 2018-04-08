from __future__ import division

import math
import numpy as np
import random


def random_erasing(img, prob=0.5,
                   scale_ratio_range=(0.02, 0.4),
                   aspect_ratio_range=(0.3, 1 / 0.3),
                   mean=[0.4914, 0.4822, 0.4465],
                   return_param=False, copy=False):
    """Select a rectangle region in an image and erase its pixels with
    ImageNet mean values.

    The size :math:`(H_{erase}, W_{erase})` and the left top coordinate
    :math:`(y_{start}, x_{start})` of the region are calculated as follows:

    + :math:`H_{erase} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\times a}}\\rfloor`
    + :math:`W_{erase} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\div a}}\\rfloor`
    + :math:`y_{start} \\sim Uniform\\{0, H - H_{erase}\\}`
    + :math:`x_{start} \\sim Uniform\\{0, W - W_{erase}\\}`
    + :math:`s \\sim Uniform(s_1, s_2)`
    + :math:`b \\sim Uniform(a_1, a_2)` and \
        :math:`a = b` or :math:`a = \\frac{1}{b}` in 50/50 probability.

    Here, :math:`s_1, s_2` are the two floats in
    :obj:`scale_ratio_interval` and :math:`a_1, a_2` are the two floats
    in :obj:`aspect_ratio_interval`.
    Also, :math:`H` and :math:`W` are the height and the width of the image.
    Note that :math:`s \\approx \\frac{H_{erase} \\times
    W_{erase}}{H \\times W}` and
    :math:`a \\approx \\frac{H_{erase}}{W_{erase}}`.
    The approximations come from flooring floats to integers.

    .. note::

        When it fails to sample a valid scale and aspect ratio for a hundred
        times, it picks values in a non-uniform way.
        If this happens, the selected scale ratio can be smaller
        than :obj:`scale_ratio_interval[0]`.

    Args:
        img (~numpy.ndarray): An image array. This is in CHW format.
        prob (float): Erasing probability.
        scale_ratio_interval (tuple of two floats): Determines
            the distribution from which a scale ratio is sampled.
        aspect_ratio_interval (tuple of two floats): Determines
            the distribution from which an aspect ratio is sampled.
        return_param (bool): Returns parameters if :obj:`True`.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns only the cropped image.

        If :obj:`return_param = True`,
        returns a tuple of erased image and :obj:`param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_slice** (*slice*): A slice used to erase a region in the input
        image. The relation below holds together with :obj:`x_slice`.
        * **x_slice** (*slice*): Similar to :obj:`y_slice`.
        * **scale_ratio** (float): :math:`s` in the description (see above).
        * **aspect_ratio** (float): :math:`a` in the description.

    """

    if random.randint(0, 1) > prob:
        _, H, W = img.shape
        scale_ratio, aspect_ratio =\
            _sample_parameters(
                (H, W), scale_ratio_range, aspect_ratio_range)

        H_crop = int(math.floor(np.sqrt(scale_ratio * H * W * aspect_ratio)))
        W_crop = int(math.floor(np.sqrt(scale_ratio * H * W / aspect_ratio)))
        y_start = random.randint(0, H - H_crop)
        x_start = random.randint(0, W - W_crop)
        y_slice = slice(y_start, y_start + H_crop)
        x_slice = slice(x_start, x_start + W_crop)

        img[:, y_slice, x_slice] = mean
    else:
        y_slice = None
        x_slice = None
        scale_ratio = None
        aspect_ratio = None

    if copy:
        img = img.copy()
    if return_param:
        params = {'y_slice': y_slice, 'x_slice': x_slice,
                  'scale_ratio': scale_ratio, 'aspect_ratio': aspect_ratio}
        return img, params
    else:
        return img


def _sample_parameters(size, scale_ratio_interval, aspect_ratio_interval):
    H, W = size
    for _ in range(100):
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
        if scale_ratio_interval[0] <= scale_ratio <= scale_ratio_max:
            return scale_ratio, aspect_ratio

    # This scale_ratio is outside the given interval when
    # scale_ratio_max < scale_ratio_interval[0].
    scale_ratio = random.uniform(
        min((scale_ratio_interval[0], scale_ratio_max)), scale_ratio_max)
    return scale_ratio, aspect_ratio
