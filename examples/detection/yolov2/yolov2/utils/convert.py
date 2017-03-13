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

import numpy as np
from chainer import serializers


def convert_darknet_model_to_npz(in_path, out_path, model=None):
    with open(in_path, 'rb') as f:
        dat = np.fromfile(f, dtype=np.float32)[4:] # skip header(4xint)

    # load model
    print("loading initial model...")
    if model is None:
        n_classes = 80
        n_boxes = 5
        from yolov2.models import YOLOv2
        model = YOLOv2(n_classes=n_classes, n_boxes=n_boxes, pretrained_model=None)
    else:
        n_classes = model.n_classes
        n_boxes = model.n_boxes

    last_out = (n_classes + 5) * n_boxes

    model.train = True
    model.finetune = False

    layers=[
        [3, 32, 3],
        [32, 64, 3], 
        [64, 128, 3], 
        [128, 64, 1], 
        [64, 128, 3], 
        [128, 256, 3], 
        [256, 128, 1], 
        [128, 256, 3], 
        [256, 512, 3], 
        [512, 256, 1], 
        [256, 512, 3], 
        [512, 256, 1], 
        [256, 512, 3], 
        [512, 1024, 3], 
        [1024, 512, 1], 
        [512, 1024, 3], 
        [1024, 512, 1], 
        [512, 1024, 3], 
        [1024, 1024, 3], 
        [1024, 1024, 3], 
        [3072, 1024, 3], 
    ]

    offset=0

    for i, l in enumerate(layers):
        in_ch = l[0]
        out_ch = l[1]
        ksize = l[2]

        # load bias(Bias.bはout_chと同じサイズ)
        txt = "model.bias%d.b.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        # load bn(BatchNormalization.gammaはout_chと同じサイズ)
        txt = "model.bn%d.gamma.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        # (BatchNormalization.avg_meanはout_chと同じサイズ)
        txt = "model.bn%d.avg_mean = dat[%d:%d]" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        # (BatchNormalization.avg_varはout_chと同じサイズ)
        txt = "model.bn%d.avg_var = dat[%d:%d]" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        # load convolution weight(Convolution2D.Wは、outch * in_ch * フィルタサイズ。これを(out_ch, in_ch, 3, 3)にreshapeする)
        txt = "model.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (i+1, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
        offset += (out_ch*in_ch*ksize*ksize)
        exec(txt)
        print(i+1, offset)

    # load last convolution weight(BiasとConvolution2Dのみロードする)
    in_ch = 1024
    out_ch = last_out
    ksize = 1

    txt = "model.bias%d.b.data = dat[%d:%d]" % (i+2, offset, offset+out_ch)
    offset += out_ch
    exec(txt)

    txt = "model.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (i+2, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
    offset += out_ch*in_ch*ksize*ksize
    exec(txt)
    print(i+2, offset)

    print("save weights file to %s" % out_path)
    serializers.save_npz(out_path, model)


if __name__ == '__main__':
    convert_darknet_model_to_npz('yolo.weights', 'yolov2_darknet.model')
