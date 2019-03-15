from __future__ import division

import numpy as np

import chainer


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
