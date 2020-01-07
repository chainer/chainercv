from __future__ import division

import chainer
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False


def _handle_four_channel_image(img, alpha):
    if alpha is None:
        raise ValueError(
            'An RGBA image is read by chainercv.utils.read_image, '
            'but the `alpha` option is not set. Please set the option so that '
            'the function knows how to handle RGBA images.'
        )
    elif alpha == 'ignore':
        img = img[:, :, :3]
    elif alpha == 'blend_with_white':
        color_channel = img[:, :, :3]
        alpha_channel = img[:, :, 3:] / 255
        img = (color_channel * alpha_channel +
               255 * np.ones_like(color_channel) * (1 - alpha_channel))
    elif alpha == 'blend_with_black':
        color_channel = img[:, :, :3]
        alpha_channel = img[:, :, 3:] / 255
        img = color_channel * alpha_channel
    return img


def _read_image_cv2(file, dtype, color, alpha):
    if color:
        if alpha is None:
            color_option = cv2.IMREAD_COLOR
        else:
            # Images with alpha channel are read as (H, W, 4) by cv2.imread.
            # Images without alpha channel are read as (H, W, 3).
            color_option = cv2.IMREAD_UNCHANGED
    else:
        color_option = cv2.IMREAD_GRAYSCALE

    if hasattr(file, 'read'):
        b = np.array(bytearray(file.read()))
        img = cv2.imdecode(b, color_option)
    else:
        img = cv2.imread(file, color_option)

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis].astype(dtype)
    else:
        # alpha channel is included
        if img.shape[-1] == 4:
            img = _handle_four_channel_image(img, alpha)
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.transpose((2, 0, 1))  # HWC -> CHW
    return img.astype(dtype)


def _read_image_pil(file, dtype, color, alpha):
    f = Image.open(file)
    try:
        if color:
            if f.mode == 'RGBA':
                img = f.convert('RGBA')
            else:
                img = f.convert('RGB')
        else:
            img = f.convert('L')
        img = np.array(img, dtype=dtype)
        if img.shape[-1] == 4:
            img = _handle_four_channel_image(
                img, alpha).astype(dtype, copy=False)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def read_image(file, dtype=np.float32, color=True, alpha=None):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    The backend used by :func:`read_image` is configured by
    :obj:`chainer.global_config.cv_read_image_backend`.
    Two backends are supported: "cv2" and "PIL".
    If this is :obj:`None`, "cv2" is used whenever "cv2" is installed,
    and "PIL" is used when "cv2" is not installed.

    Args:
        file (string or file-like object): A path of image file or
            a file-like object of image.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.
        alpha (None or {'ignore', 'blend_with_white', 'blend_with_black'}): \
            Choose how RGBA images are handled. By default, an error is raised.
            Here are the other possible behaviors:

            * `'ignore'`: Ignore alpha channel.
            * `'blend_with_white'`: Blend RGB image multiplied by alpha on \
                 a white image.
            * `'blend_with_black'`: Blend RGB image multiplied by alpha on \
                 a black image.

    Returns:
        ~numpy.ndarray: An image.
    """
    if chainer.config.cv_read_image_backend is None:
        if _cv2_available:
            return _read_image_cv2(file, dtype, color, alpha)
        else:
            return _read_image_pil(file, dtype, color, alpha)
    elif chainer.config.cv_read_image_backend == 'cv2':
        if not _cv2_available:
            raise ValueError('cv2 is not installed even though '
                             'chainer.config.cv_read_image_backend == \'cv2\'')
        return _read_image_cv2(file, dtype, color, alpha)
    elif chainer.config.cv_read_image_backend == 'PIL':
        return _read_image_pil(file, dtype, color, alpha)
    else:
        raise ValueError('chainer.config.cv_read_image_backend should be '
                         'either "cv2" or "PIL".')
