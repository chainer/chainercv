#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.pyplot as plot

import chainer
from chainercv.datasets import cityscapes_label_colors
from chainercv.datasets import cityscapes_label_names
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import PSPNet
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_fn', '-f', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--scales', '-s', type=float, nargs='*', default=None)
    parser.add_argument('--model', '-m', type=str,
                        choices=['voc2012', 'cityscapes'])
    args = parser.parse_args()

    chainer.config.train = False

    if args.model == 'voc2012':
        model = PSPNet(pretrained_model='voc2012')
        labels = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors
    elif args.model == 'cityscapes':
        model = PSPNet(pretrained_model='cityscapes')
        labels = cityscapes_label_names
        colors = cityscapes_label_colors

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    img = read_image(args.img_fn)
    pred = model.predict([img])

    # Save the result image
    ax = vis_image(img)
    _, legend_handles = vis_label(pred, labels, colors, alpha=1.0, ax=ax)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.)
    base = os.path.splitext(os.path.basename(args.img_fn))[0]
    plot.savefig('predict_{}.png'.format(base), bbox_inches='tight', dpi=400)
