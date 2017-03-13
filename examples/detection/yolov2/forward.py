#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
#
# Original work by:
# --------------------------------------------------------
# YOLOv2
# Copyright (c) 2017 leetenki
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/leetenki/YOLOv2
# --------------------------------------------------------

import os
import mock
import numpy as np

import chainer
import chainer.functions as F

from chainercv.datasets import VOCDetectionDataset
from chainercv.datasets import TransformDataset
from chainercv.extensions import DetectionVisReport
import chainercv.transforms as T

from yolov2.models import YOLOv2, YOLOv2Predictor
from yolov2.utils import Box, nms


def _min_max_shape(in_shape, soft_min, hard_max):
    h, w = in_shape
    min_edge = np.minimum(w, h)
    if min_edge < soft_min:
        w = w * soft_min / min_edge
        h = h * soft_min / min_edge
    max_edge = np.maximum(w, h)
    if max_edge > hard_max:
        w = w * hard_max / max_edge
        h = h * hard_max / max_edge

    w = int(w / 32 + round(w % 32 / 32)) * 32
    h = int(h / 32 + round(h % 32 / 32)) * 32
    return (h, w)


if __name__ == '__main__':
    test_data = VOCDetectionDataset(mode='train', year='2007')

    def transform(in_data):
        img, bbox = in_data
        in_shape = img.shape[1:]
        img /= 255.0
        out_shape = _min_max_shape(in_shape, 320, 448)
        img = T.resize(img, out_shape)
        bbox = T.resize_bbox(bbox, in_shape, out_shape)
        return img, bbox

    test_data = TransformDataset(test_data, transform)
    n_classes = len(test_data.labels) - 1 # remove background
    n_boxes = 5
    iou_threshold = 0.5
    detection_threshold = 0.5

    model = YOLOv2(n_classes=n_classes, n_boxes=n_boxes, pretrained_model='voc')
    model.train = False
    model.finetune = False
    predictor = YOLOv2Predictor(model)

    trainer = mock.MagicMock()
    trainer.out = 'result'
    trainer.updater.iteration = 0

    if not os.path.exists(trainer.out):
        os.makedirs(trainer.out)

    def predict_bboxes(in_data, bboxes=None):
        img_h, img_w = in_data.shape[2:]
        x, y, w, h, conf, prob = predictor.predict(in_data)
        _, _, _, grid_h, grid_w = x.shape

        x = F.reshape(x, (n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (n_boxes, n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > detection_threshold

        results = []
        for i in range(detected_indices.sum()):
            results.append({
                "class_id": prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()+1, # shift for background class
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box"  : Box(
                            x[detected_indices][i]*img_w,
                            y[detected_indices][i]*img_h,
                            w[detected_indices][i]*img_w,
                            h[detected_indices][i]*img_h).crop_region(img_h, img_w)
            })

        results = nms(results, iou_threshold)
        n_bboxes = len(results)
        bboxes = np.zeros((n_bboxes, 5), dtype=np.float32)
        for i in range(n_bboxes):
            box = results[i]["box"]
            bboxes[i] = [box.int_left_top()[0],
                         box.int_left_top()[1],
                         box.int_right_bottom()[0],
                         box.int_right_bottom()[1],
                         results[i]["class_id"]]

        return bboxes[None],

    def vis_transform(in_data):
        img, bboxes = in_data
        img *= 255.0
        img, = T.chw_to_pil_image_tuple((img,))
        return img, bboxes

    extension = DetectionVisReport(
        range(5,10),
        test_data,
        model,
        predict_func=predict_bboxes,
        vis_transform=vis_transform)
    extension(trainer)
