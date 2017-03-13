#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mofidied by:
# Copyright (c) 2017 Yuki Furuta
#
# Original work by:
# --------------------------------------------------------
# YOLOv2
# Copyright (c) 2017 leetenki
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/leetenki/YOLOv2
# --------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import

import chainer.functions as F
import numpy as np
import cv2
from chainer import Variable

def print_cnn_info(name, link, shape_before, shape_after, time):
    n_stride = (
        int((shape_before[2] + link.pad[0] * 2 - link.ksize) / link.stride[0]) + 1,
        int((shape_before[3] + link.pad[1] * 2 - link.ksize) / link.stride[1]) + 1
    )

    cost = n_stride[0] * n_stride[1] * shape_before[1] * link.ksize * link.ksize * link.out_channels

    print('%s(%d × %d, stride=%d, pad=%d) (%d x %d x %d) -> (%d x %d x %d) (cost=%d): %.6f[sec]' % 
        (
            name, link.W.shape[2], link.W.shape[3], link.stride[0], link.pad[0],
            shape_before[2], shape_before[3], shape_before[1], shape_after[2], shape_after[3], shape_after[1],
            cost, time
        )
    )

    return cost

def print_pooling_info(name, filter_size, stride, pad, shape_before, shape_after, time):
    n_stride = (
        int((shape_before[2] - filter_size) / stride) + 1,
        int((shape_before[3] - filter_size) / stride) + 1
    )
    cost = n_stride[0] * n_stride[1] * shape_before[1] * filter_size * filter_size * shape_after[1]

    print('%s(%d × %d, stride=%d, pad=%d) (%d x %d x %d) -> (%d x %d x %d) (cost=%d): %.6f[sec]' % 
        (name, filter_size, filter_size, stride, pad, shape_before[2], shape_before[3], shape_before[1], shape_after[2], shape_after[3], shape_after[1], cost, time)
    )

    return cost

def print_fc_info(name, link, time):
    import pdb
    cost = link.W.shape[0] * link.W.shape[1]
    print('%s %d -> %d (cost = %d): %.6f[sec]' % (name, link.W.shape[1], link.W.shape[0], cost, time))

    return cost

class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def int_left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x - half_width)), int(round(self.y - half_height)))

    def left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x - half_width, self.y - half_height]

    def int_right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x + half_width)), int(round(self.y + half_height)))

    def right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x + half_width, self.y + half_height]

    def crop_region(self, h, w):
        left, top = self.left_top()
        right, bottom = self.right_bottom()
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        self.w = right - left
        self.h = bottom - top
        self.x = (right + left) / 2
        self.y = (bottom + top) / 2
        return self

def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

def multi_overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = F.maximum(x1 - len1_half, x2 - len2_half)
    right = F.minimum(x1 + len1_half, x2 + len2_half)

    return right - left

def box_intersection(a, b):
    """Intersection of two boxes"""
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def multi_box_intersection(a, b):
    w = multi_overlap(a.x, a.w, b.x, b.w)
    h = multi_overlap(a.y, a.h, b.y, b.h)
    zeros = Variable(np.zeros(w.shape, dtype=w.data.dtype))
    zeros.to_gpu()

    w = F.maximum(w, zeros)
    h = F.maximum(h, zeros)

    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

def multi_box_union(a, b):
    i = multi_box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def multi_box_iou(a, b):
    return multi_box_intersection(a, b) / multi_box_union(a, b)

def nms(predicted_results, iou_thresh):
    nms_results = []
    for i in range(len(predicted_results)):
        overlapped = False
        for j in range(i+1, len(predicted_results)):
            if box_iou(predicted_results[i]["box"], predicted_results[j]["box"]) > iou_thresh:
                overlapped = True
                if predicted_results[i]["objectness"] > predicted_results[j]["objectness"]:
                    temp = predicted_results[i]
                    predicted_results[i] = predicted_results[j]
                    predicted_results[j] = temp
        if not overlapped:
            nms_results.append(predicted_results[i])
    return nms_results
