#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.pyplot as plot
import numpy as np
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.transforms import center_crop
from chainercv.transforms import scale
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_label
from datasets import cityscapes_label_names
from datasets import cityscapes_label_colors

import chainer
from chainer import serializers
from pspnet import PSPNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_fn', '-f', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--fit', action='store_true', default=False)
    parser.add_argument(
        '--model', type=str, choices=['VOC', 'Cityscapes', 'ADE20K'])
    args = parser.parse_args()

    if args.model == 'VOC':
        n_class = 21
        n_blocks = [3, 4, 23, 3]
        feat_size = 60
        mid_stride = True
        param_fn = 'weights/pspnet101_VOC2012_473_reference.chainer'
        crop_size = 473
        labels = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors
    elif args.model == 'Cityscapes':
        n_class = 19
        n_blocks = [3, 4, 23, 3]
        feat_size = 90
        mid_stride = True
        param_fn = 'weights/pspnet101_cityscapes_713_reference.chainer'
        crop_size = 713
        labels = cityscapes_label_names
        colors = cityscapes_label_colors
    elif args.model == 'ADE20K':
        n_class = 150
        n_blocks = [3, 4, 6, 3]
        feat_size = 60
        mid_stride = False
        param_fn = 'weights/pspnet101_ADE20K_473_reference.chainer'
        crop_size = 473

    chainer.config.train = False
    model = PSPNet(n_class, n_blocks, feat_size, mid_stride=mid_stride)
    serializers.load_npz(param_fn, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    img = read_image(args.img_fn)
    if args.fit:
        img = scale(img, crop_size, fit_short=True)
        img = center_crop(img, (crop_size, crop_size))
    img = img[::-1]  # Convert to BGR order

    # Mean values in BGR order
    mean = np.array([103.939, 116.779, 123.68])

    # Mean subtraction
    img -= mean[:, None, None]

    # Inference
    pred = model.predict([img])[0]
    print('Predicted label IDs:', np.unique(pred))

    ax = vis_image(img)
    _, legend_handles = vis_label(pred, labels, colors, alpha=1.0, ax=ax)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.)
    base = os.path.splitext(os.path.basename(args.img_fn))[0]
    plot.savefig('predict_{}.png'.format(base), bbox_inches='tight')
