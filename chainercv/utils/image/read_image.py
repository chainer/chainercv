import chainer
import numpy as np
from PIL import Image
import warnings

try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False


def _read_image_cv2(path, dtype, color):
    if color:
        color_option = cv2.IMREAD_COLOR
    else:
        color_option = cv2.IMREAD_GRAYSCALE

    img = cv2.imread(path, color_option)

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis].astype(dtype)
    else:
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.transpose((2, 0, 1))  # HWC -> CHW
    return img.astype(dtype)


def _read_image_pil(path, dtype, color):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('L')
        img = np.asarray(img, dtype=dtype)
        img.flags.writeable = True
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    The backend used by :func:`read_image` is configured by
    :obj:`chainer.global_config.cv_read_image_backend`.
    Two backends are supported: "cv2" and "PIL".

    Args:
        path (string): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """
    if chainer.config.cv_read_image_backend == 'cv2':
        if _cv2_available:
            return _read_image_cv2(path, dtype, color)
        else:
            warnings.warn(
                'Although `chainer.config.cv_read_image_backend == "cv2"`, '
                'cv2 is not found. As a fallback option, read_image uses '
                'PIL. Either install cv2 or set '
                '`chainer.global_config.cv_read_image_backend = "PIL"` '
                'to suppress this warning.')
            return _read_image_pil(path, dtype, color)
    elif chainer.config.cv_read_image_backend == 'PIL':
        return _read_image_pil(path, dtype, color)
    else:
        raise ValueError('chainer.config.cv_read_image_backend should be '
                         'either "cv2" or "PIL".')
