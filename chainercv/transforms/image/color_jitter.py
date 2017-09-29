import numpy as np
import random


def _grayscale(img):
    out = np.zeros_like(img)
    out[:] = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return out


def _blend(img_a, img_b, alpha):
    return alpha * img_a + (1 - alpha) * img_b


def _brightness(img, var):
    alpha = 1 + np.random.uniform(-var, var)
    return _blend(img, np.zeros_like(img), alpha), alpha


def _contrast(img, var):
    gray = _grayscale(img)
    gray.fill(gray[0].mean())

    alpha = 1 + np.random.uniform(-var, var)
    return _blend(img, gray, alpha), alpha


def _saturation(img, var):
    gray = _grayscale(img)

    alpha = 1 + np.random.uniform(-var, var)
    return _blend(img, gray, alpha), alpha


def color_jitter(img, brightness_var=0.4, contrast_var=0.4,
                 saturation_var=0.4, return_param=False):
    """Data augmentation on brightness, contrast and saturation.

    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW and RGB format.
        brightness_var (float): Alpha for brightness is sampled from
            :obj:`unif(-brightness_var, brightness_var)`. The default
            value is 0.4.
        contrast_var (float): Alpha for contrast is sampled from
            :obj:`unif(-contrast_var, contrast_var)`. The default
            value is 0.4.
        saturation_var (float): Alpha for contrast is sampled from
            :obj:`unif(-saturation_var, saturation_var)`. The default
            value is 0.4.
        return_param (bool): Returns parameters if :obj:`True`.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an color jittered image.

        If :obj:`return_param = True`, returns a tuple of an array and a
        dictionary :obj:`param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **order** (*list of strings*): List containing three strings: \
            :obj:`'brightness'`, :obj:`'contrast'` and :obj:`'saturation'`. \
            They are ordered according to the order in which the data \
            augmentation functions are applied.
        * **brightness_alpha** (*float*): Alpha used for brightness \
            data augmentation.
        * **contrast_alpha** (*float*): Alpha used for contrast \
            data augmentation.
        * **saturation_alpha** (*float*): Alpha used for saturation \
            data augmentation.

    """
    funcs = []
    if brightness_var > 0:
        funcs.append(('brightness', lambda x: _brightness(x, brightness_var)))
    if contrast_var > 0:
        funcs.append(('contrast', lambda x: _contrast(x, contrast_var)))
    if saturation_var > 0:
        funcs.append(('saturation', lambda x: _saturation(x, saturation_var)))
    random.shuffle(funcs)

    params = {'order': [key for key, val in funcs],
              'brightness_alpha': 1,
              'contrast_alpha': 1,
              'saturation_alpha': 1}
    for key, func in funcs:
        img, alpha = func(img)
        params[key + '_alpha'] = alpha
    img = np.minimum(np.maximum(img, 0), 255)
    if return_param:
        return img, params
    else:
        return img
