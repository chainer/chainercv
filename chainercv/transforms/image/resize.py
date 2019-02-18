import chainer
import numpy as np
import PIL
import warnings


try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False


def _resize_cv2(img, size, interpolation):
    img = img.transpose((1, 2, 0))
    if interpolation == PIL.Image.NEAREST:
        cv_interpolation = cv2.INTER_NEAREST
    elif interpolation == PIL.Image.BILINEAR:
        cv_interpolation = cv2.INTER_LINEAR
    elif interpolation == PIL.Image.BICUBIC:
        cv_interpolation = cv2.INTER_CUBIC
    elif interpolation == PIL.Image.LANCZOS:
        cv_interpolation = cv2.INTER_LANCZOS4
    H, W = size
    img = cv2.resize(img, dsize=(W, H), interpolation=cv_interpolation)

    # If input is a grayscale image, cv2 returns a two-dimentional array.
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img.transpose((2, 0, 1))


def _resize_pil(img, size, interpolation):
    C = img.shape[0]
    H, W = size
    out = np.empty((C, H, W), dtype=img.dtype)
    for ch, out_ch in zip(img, out):
        ch = PIL.Image.fromarray(ch, mode='F')
        out_ch[:] = ch.resize((W, H), resample=interpolation)
    return out


def resize(img, size, interpolation=PIL.Image.BILINEAR):
    """Resize image to match the given shape.

    The backend used by :func:`resize` is configured by
    :obj:`chainer.global_config.cv_resize_backend`.
    Two backends are supported: "cv2" and "PIL".

    Args:
        img (~numpy.ndarray): An array to be transformed.
            This is in CHW format and the type should be :obj:`numpy.float32`.
        size (tuple): This is a tuple of length 2. Its elements are
            ordered as (height, width).
        interpolation (int): Determines sampling strategy. This is one of
            :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BILINEAR`,
            :obj:`PIL.Image.BICUBIC`, :obj:`PIL.Image.LANCZOS`.
            Bilinear interpolation is the default strategy.

    Returns:
        ~numpy.ndarray: A resize array in CHW format.

    """
    if len(img) == 0:
        assert len(size) == 2
        return np.zeros((0,) + size, dtype=img.dtype)

    if chainer.config.cv_resize_backend == 'cv2':
        if _cv2_available:
            return _resize_cv2(img, size, interpolation)
        else:
            warnings.warn(
                'Although `chainer.config.cv_resize_backend == "cv2"`, '
                'cv2 is not found. As a fallback option, resize uses '
                'PIL. Either install cv2 or set '
                '`chainer.global_config.cv_resize_backend = "PIL"` to '
                'suppress this warning.')
            return _resize_pil(img, size, interpolation)
    elif chainer.config.cv_resize_backend == 'PIL':
        return _resize_pil(img, size, interpolation)
    else:
        raise ValueError('chainer.config.cv_resize_backend should be '
                         'either "cv2" or "PIL".')
