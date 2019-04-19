from __future__ import division
import numpy as np
import PIL

import chainer

try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False


def _rotate_cv2(img, angle, expand, fill, interpolation):
    if interpolation == PIL.Image.NEAREST:
        cv_interpolation = cv2.INTER_NEAREST
    elif interpolation == PIL.Image.BILINEAR:
        cv_interpolation = cv2.INTER_LINEAR
    elif interpolation == PIL.Image.BICUBIC:
        cv_interpolation = cv2.INTER_CUBIC

    _, H, W = img.shape
    affine_mat = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1)
    # Logic borrowed from Pillow
    if expand:
        # calculate output size
        yy = []
        xx = []
        for y, x in ((0, 0), (H, 0), (H, W), (0, W)):
            yy.append(
                affine_mat[1, 0] * x + affine_mat[1, 1] * y + affine_mat[1, 2])
            xx.append(
                affine_mat[0, 0] * x + affine_mat[0, 1] * y + affine_mat[0, 2])
        out_H = int(np.ceil(max(yy)) - np.floor(min(yy)))
        out_W = int(np.ceil(max(xx)) - np.floor(min(xx)))

        affine_mat[1][2] += out_H / 2 - H / 2
        affine_mat[0][2] += out_W / 2 - W / 2
    else:
        out_H = H
        out_W = W

    img = img.transpose((1, 2, 0))
    img = cv2.warpAffine(
        img, affine_mat, (out_W, out_H), flags=cv_interpolation,
        borderValue=fill)
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.transpose((2, 0, 1))
    return img


def _rotate_pil(img, angle, expand, fill, interpolation):
    out = []
    for ch in img:
        ch = PIL.Image.fromarray(ch, mode='F')
        out.append(np.array(
            ch.rotate(
                angle, expand=expand,
                fillcolor=fill, resample=interpolation)))
    out = np.stack(out)
    if np.issubdtype(img.dtype, np.integer):
        out = np.round(out)
    return out.astype(img.dtype)


def rotate(img, angle, expand=True, fill=0, interpolation=PIL.Image.BILINEAR):
    """Rotate images by degrees.

    The backend used by :func:`rotate` is configured by
    :obj:`chainer.global_config.cv_rotate_backend`.
    Two backends are supported: "cv2" and "PIL".
    If this is :obj:`None`, "cv2" is used whenever "cv2" is installed,
    and "PIL" is used when "cv2" is not installed.

    Args:
        img (~numpy.ndarray): An arrays that get rotated. This is in
            CHW format.
        angle (float): Counter clock-wise rotation angle (degree).
        expand (bool): The output shaped is adapted or not.
            If :obj:`True`, the input image is contained complete in
            the output.
        fill (float): The value used for pixels outside the boundaries.
        interpolation (int): Determines sampling strategy. This is one of
            :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BILINEAR`,
            :obj:`PIL.Image.BICUBIC`.
            Bilinear interpolation is the default strategy.

    Returns:
        ~numpy.ndarray:
        returns an array :obj:`out_img` that is the result of rotation.

    """
    if chainer.config.cv_rotate_backend is None:
        if _cv2_available:
            return _rotate_cv2(img, angle, expand, fill, interpolation)
        else:
            return _rotate_pil(img, angle, expand, fill, interpolation)
    elif chainer.config.cv_rotate_backend == 'cv2':
        if not _cv2_available:
            raise ValueError('cv2 is not installed even though '
                             'chainer.config.cv_rotate_backend == \'cv2\'')
        return _rotate_cv2(img, angle, expand, fill, interpolation)
    elif chainer.config.cv_rotate_backend == 'PIL':
        return _rotate_pil(img, angle, expand, fill, interpolation)
    else:
        raise ValueError('chainer.config.cv_rotate_backend should be '
                         'either "cv2" or "PIL".')
