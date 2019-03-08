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
    bbox = _integerize_bbox(_expand_boxes(bbox, cv2_expand_scale))

    segm = []
    if index is None:
        index = np.arange(len(bbox))
    else:
        index = chainer.backends.cuda.to_cpu(index)

    for i, bb in zip(index, bbox):
        y_min = max(bb[0], 0)
        x_min = max(bb[1], 0)
        y_max = max(min(bb[2], H), 0)
        x_max = max(min(bb[3], W), 0)
        if y_max - y_min == 0 or x_max - x_min == 0:
            segm.append(np.zeros((segm_size, segm_size), dtype=np.float32))
            continue

        bb_height = bb[2] - bb[0]
        bb_width = bb[3] - bb[1]
        cropped_m = np.zeros((bb_height, bb_width), dtype=np.bool)

        y_offset = y_min - bb[0]
        x_offset = x_min - bb[1]
        cropped_m[y_offset:y_offset + y_max - y_min,
                  x_offset:x_offset + x_max - x_min] =\
            chainer.backends.cuda.to_cpu(mask[i, y_min:y_max, x_min:x_max])

        sgm = transforms.resize(
            cropped_m[None].astype(np.float32),
            (padded_segm_size, padded_segm_size))[0].astype(np.int32)
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

    bbox = _integerize_bbox(_expand_boxes(bbox, cv2_expand_scale))
    for i, (bb, sgm) in enumerate(zip(bbox, segm)):
        padded_mask[1:-1, 1:-1] = sgm

        bb_height = bb[2] - bb[0]
        bb_width = bb[3] - bb[1]
        if bb_height == 0 or bb_width == 0:
            continue

        crop_mask = transforms.resize(padded_mask[None], (bb_width, bb_height))[0]
        crop_mask = crop_mask > 0.5

        y_min = max(bb[0], 0)
        x_min = max(bb[1], 0)
        y_max = max(min(bb[2], H), 0)
        x_max = max(min(bb[3], W), 0)
        y_offset = y_min - bb[0]
        x_offset = x_min - bb[1]
        mask[i, y_min:y_max, x_min:x_max] = crop_mask[
            y_offset:y_offset + y_max - y_min,
            x_offset:x_offset + x_max - x_min]
    return mask


def _integerize_bbox(bbox):
    return np.round(bbox).astype(np.int32)


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


def point_to_roi_points(
        point, visible, bbox, point_map_size):
    xp = chainer.backends.cuda.get_array_module(point)

    R, K, _ = point.shape

    roi_point = xp.zeros((len(bbox), K, 2))
    roi_visible = xp.zeros((len(bbox), K), dtype=np.bool)

    offset_y = bbox[:, 0]
    offset_x = bbox[:, 1]
    scale_y = point_map_size / (bbox[:, 2] - bbox[:, 0])
    scale_x = point_map_size / (bbox[:, 3] - bbox[:, 1])

    for k in range(K):
        y_boundary_index = xp.where(point[:, k, 0] == bbox[:, 2])[0]
        x_boundary_index = xp.where(point[:, k, 1] == bbox[:, 3])[0]

        ys = (point[:, k, 0] - offset_y) * scale_y
        ys = xp.floor(ys)
        if len(y_boundary_index) > 0:
            ys[y_boundary_index] = point_map_size - 1
        xs = (point[:, k, 1] - offset_x) * scale_x
        xs = xp.floor(xs)
        if len(x_boundary_index) > 0:
            xs[x_boundary_index] = point_map_size - 1

        valid = xp.logical_and(
            xp.logical_and(
                xp.logical_and(ys >= 0, xs >= 0),
                xp.logical_and(ys < point_map_size, xs < point_map_size)),
            visible[:, k])

        roi_point[:, k, 0] = ys
        roi_point[:, k, 1] = xs
        roi_visible[:, k] = valid
    return roi_point, roi_visible


def within_bbox(point, bbox):
    y_within = (point[:, :, 0] >= bbox[:, 0][:, None]) & (
        point[:, :, 0] <= bbox[:, 2][:, None])
    x_within = (point[:, :, 1] >= bbox[:, 1][:, None]) & (
        point[:, :, 1] <= bbox[:, 3][:, None])
    return y_within & x_within
