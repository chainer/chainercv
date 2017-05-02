# Mofidied by:
# Copyright (c) 2016 Shunta Saito

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

import numpy as np

from chainer import cuda
from chainer.cuda import get_array_module


def bbox_transform(ex_rois, gt_rois):
    xp = get_array_module(ex_rois)

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = xp.log(gt_widths / ex_widths)
    targets_dh = xp.log(gt_heights / ex_heights)

    targets = xp.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas, gpu=-1):
    if gpu >= 0:
        with cuda.Device(gpu):
            return _bbox_transform_inv(boxes, deltas)
    else:
        return _bbox_transform_inv(boxes, deltas)


def _bbox_transform_inv(boxes, deltas):
    xp = get_array_module(boxes)

    if boxes.shape[0] == 0:
        return xp.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, xp.newaxis] + ctr_x[:, xp.newaxis]
    pred_ctr_y = dy * heights[:, xp.newaxis] + ctr_y[:, xp.newaxis]
    pred_w = xp.exp(dw) * widths[:, xp.newaxis]
    pred_h = xp.exp(dh) * heights[:, xp.newaxis]

    pred_boxes = xp.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, img_size, gpu=-1):
    if gpu >= 0:
        with cuda.Device(gpu):
            return _clip_boxes(boxes, img_size)
    else:
        return _clip_boxes(boxes, img_size)


def _clip_boxes(boxes, img_size):
    """Clip boxes to image boundaries."""
    xp = get_array_module(boxes)
    W, H = img_size

    # x1 >= 0
    boxes[:, 0::4] = xp.maximum(xp.minimum(boxes[:, 0::4], W - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = xp.maximum(xp.minimum(boxes[:, 1::4], H - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = xp.maximum(xp.minimum(boxes[:, 2::4], W - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = xp.maximum(xp.minimum(boxes[:, 3::4], H - 1), 0)
    return boxes


def keep_inside(anchors, W, H):
    """Calc indicies of anchors which are inside of the image size.

    Calc indicies of anchors which are located completely inside of the image
    whose size is speficied by img_info ((height, width, scale)-shaped array).
    """
    with cuda.get_device_from_array(anchors) as d:
        xp = cuda.get_array_module(anchors)
        if d.id >= 0:
            img_info = cuda.to_gpu(img_info, d)
            assert anchors.device == img_info.device

        inds_inside = xp.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] < W) &  # width
            (anchors[:, 3] < H)  # height
        )[0]
        return inds_inside, anchors[inds_inside]


def get_bbox_regression_label(bbox, label, n_class, bbox_inside_weight_coeff):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation
    used by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights

    """
    n_bbox = label.shape[0]
    bbox_target = np.zeros((n_bbox, 4 * n_class), dtype=np.float32)
    bbox_inside_weight = np.zeros_like(bbox_target)
    inds = np.where(label > 0)[0]
    for ind in inds:
        cls = int(label[ind])
        start = int(4 * cls)
        end = int(start + 4)
        bbox_target[ind, start:end] = bbox[ind]
        bbox_inside_weight[ind, start:end] = bbox_inside_weight_coeff
    return bbox_target, bbox_inside_weight
