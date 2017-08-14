#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import chainer
import matplotlib.pyplot as plot
import numpy as np
from chainer import serializers
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_label
from skimage import io

from datasets import cityscapes_label_colors
from datasets import cityscapes_label_names
from datasets import cityscapes_labels
from evaluate import inference
from evaluate import preprocess
from chainercv.links import PSPNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_fn', '-f', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--scales', '-s', type=float, nargs='*', default=None)
    parser.add_argument(
        '--model', '-m', type=str, choices=['VOC', 'Cityscapes', 'ADE20K'])
    args = parser.parse_args()

    chainer.config.stride_rate = args.stride_rate
    chainer.config.save_test_image = args.save_test_image

    if args.model == 'VOC':
        model = PSPNet(pretrained_model='voc2012')
        labels = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors
    elif args.model == 'Cityscapes':
        n_class = 19
        n_blocks = [3, 4, 23, 3]
        feat_size = 90
        mid_stride = True
        param_fn = 'weights/pspnet101_cityscapes_713_reference.chainer'
        base_size = 2048
        crop_size = 713
        labels = cityscapes_label_names
        colors = cityscapes_label_colors
    elif args.model == 'ADE20K':
        n_class = 150
        n_blocks = [3, 4, 6, 3]
        feat_size = 60
        mid_stride = False
        param_fn = 'weights/pspnet101_ADE20K_473_reference.chainer'
        base_size = 512
        crop_size = 473

    chainer.config.train = False
    model = PSPNet(n_class, n_blocks, feat_size, mid_stride=mid_stride)
    serializers.load_npz(param_fn, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    img = preprocess(read_image(args.img_fn))

    # Inference
    pred = inference(
        model, n_class, base_size, crop_size, img, args.scales)

    # Save the result image
    ax = vis_image(img)
    _, legend_handles = vis_label(pred, labels, colors, alpha=1.0, ax=ax)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.)
    base = os.path.splitext(os.path.basename(args.img_fn))[0]
    plot.savefig('predict_{}.png'.format(base), bbox_inches='tight', dpi=400)

    if args.model == 'Cityscapes':
        label_out = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
        for label in cityscapes_labels:
            label_out[np.where(pred == label.trainId)] = label.color
        io.imsave(
            'predict_{}_color({}).png'.format(base, args.scales), label_out)
