# Mofidied by:
# Copyright (c) 2016 Shunta Saito

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

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


def clip_boxes(boxes, im_shape, gpu=-1):
    if gpu >= 0:
        with cuda.Device(gpu):
            return _clip_boxes(boxes, im_shape)
    else:
        return _clip_boxes(boxes, im_shape)


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    xp = get_array_module(boxes)

    # x1 >= 0
    boxes[:, 0::4] = xp.maximum(xp.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = xp.maximum(xp.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = xp.maximum(xp.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = xp.maximum(xp.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
