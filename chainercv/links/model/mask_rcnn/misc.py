from __future__ import division

import cv2
import numpy as np

import chainer

from chainercv import transforms


def scale_img(img, min_size, max_size):
    """Process image."""
    _, H, W = img.shape
    scale = min_size / min(H, W)
    if scale * max(H, W) > max_size:
        scale = max_size / max(H, W)
    H, W = int(H * scale), int(W * scale)
    img = transforms.resize(img, (H, W))
    return img, scale


def mask_to_segm(mask, bbox, segm_size, index=None, pad=1):
    """Crop and resize mask.

    Args:
        mask (~numpy.ndarray): See below.
        bbox (~numpy.ndarray): See below.
        segm_size (int): The size of segm :math:`S`.
        index (~numpy.ndarray): See below. :math:`R = N` when
            :obj:`index` is :obj:`None`.
        pad (int): The amount of padding used for bbox.

    Returns:
        ~numpy.ndarray: See below.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`mask`, ":math:`(N, H, W)`", :obj:`bool`, --
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`index` (optional), ":math:`(R,)`", :obj:`int32`, --
        :obj:`segms` (output), ":math:`(R, S, S)`", :obj:`float32`, \
        ":math:`[0, 1]`"

    """
    _, H, W = mask.shape
    bbox = chainer.backends.cuda.to_cpu(bbox)
    padded_segm_size = segm_size + pad * 2
    cv2_expand_scale = padded_segm_size / segm_size
    bbox = _expand_boxes(bbox, cv2_expand_scale).astype(np.int32)

    segm = []
    if index is None:
        index = np.arange(len(index))
    else:
        index = chainer.backends.cuda.to_cpu(index)

    for i, bb in zip(index, bbox):
        y_min = max(bb[0], 0)
        x_min = max(bb[1], 0)
        y_max = min(bb[2] + 1, H)
        x_max = min(bb[3] + 1, W)
        cropped_m = mask[i, y_min:y_max, x_min:x_max]
        cropped_m = chainer.backends.cuda.to_cpu(cropped_m)
        if cropped_m.shape[0] <= 1 or cropped_m.shape[1] <= 1:
            segm.append(np.zeros((segm_size, segm_size), dtype=np.float32))
            continue

        sgm = transforms.resize(
            cropped_m[None].astype(np.float32),
            (padded_segm_size, padded_segm_size))[0]
        segm.append(sgm[pad:-pad, pad:-pad])

    return np.array(segm, dtype=np.float32)


def segm_to_mask(segm, bbox, size, pad=1):
    """Recover mask from cropped and resized mask.

    Args:
        segm (~numpy.ndarray): See below.
        bbox (~numpy.ndarray): See below.
        size (tuple): This is a tuple of length 2. Its elements are
            ordered as (height, width).
        pad (int): The amount of padding used for bbox.

    Returns:
        ~numpy.ndarray: See below.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`segm`, ":math:`(R, S, S)`", :obj:`float32`, --
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`mask` (output), ":math:`(R, H, W)`", :obj:`bool`, --

    """
    H, W = size
    _, segm_size, _ = segm.shape

    mask = np.zeros((len(bbox), H, W), dtype=np.bool)

    # To work around an issue with cv2.resize (it seems to automatically
    # pad with repeated border values), we manually zero-pad the masks by 1
    # pixel prior to resizing back to the original image resolution.
    # This prevents "top hat" artifacts. We therefore need to expand
    # the reference boxes by an appropriate factor.
    cv2_expand_scale = (segm_size + pad * 2) / segm_size
    padded_mask = np.zeros(
        (segm_size + pad * 2, segm_size + pad * 2), dtype=np.float32)

    bbox = _expand_boxes(bbox, cv2_expand_scale)
    for i, (bb, sgm) in enumerate(zip(bbox, segm)):
        bb = bb.astype(np.int32)
        padded_mask[1:-1, 1:-1] = sgm

        bb_height = np.maximum(bb[2] - bb[0] + 1, 1)
        bb_width = np.maximum(bb[3] - bb[1] + 1, 1)

        crop_mask = cv2.resize(padded_mask, (bb_width, bb_height))
        crop_mask = crop_mask > 0.5

        y_min = max(bb[0], 0)
        x_min = max(bb[1], 0)
        y_max = min(bb[2] + 1, H)
        x_max = min(bb[3] + 1, W)
        mask[i, y_min:y_max, x_min:x_max] = crop_mask[
            (y_min - bb[0]):(y_max - bb[0]),
            (x_min - bb[1]):(x_max - bb[1])]
    return mask


def _expand_boxes(bbox, scale):
    """Expand an array of boxes by a given scale."""
    xp = chainer.backends.cuda.get_array_module(bbox)

    h_half = (bbox[:, 2] - bbox[:, 0]) * .5
    w_half = (bbox[:, 3] - bbox[:, 1]) * .5
    y_c = (bbox[:, 2] + bbox[:, 0]) * .5
    x_c = (bbox[:, 3] + bbox[:, 1]) * .5

    h_half *= scale
    w_half *= scale

    expanded_bbox = xp.zeros(bbox.shape)
    expanded_bbox[:, 0] = y_c - h_half
    expanded_bbox[:, 1] = x_c - w_half
    expanded_bbox[:, 2] = y_c + h_half
    expanded_bbox[:, 3] = x_c + w_half

    return expanded_bbox
