#!/bin/bash

if [ ! -f weights/pspnet101_cityscapes_713_reference.chainer ]; then
    curl -L https://github.com/mitmul/chainer-pspnet/releases/download/PSPNet_reference_weights/pspnet101_cityscapes_713_reference.chainer -o weights/pspnet101_cityscapes_713_reference.chainer
fi

if [ ! -f weights/pspnet101_VOC2012_473_reference.chainer ]; then
    curl -L https://github.com/mitmul/chainer-pspnet/releases/download/PSPNet_reference_weights/pspnet101_VOC2012_473_reference.chainer -o weights/pspnet101_VOC2012_473_reference.chainer
fi

if [ ! -f weights/pspnet50_ADE20K_473_reference.chainer ]; then
    curl -L https://github.com/mitmul/chainer-pspnet/releases/download/PSPNet_reference_weights/pspnet50_ADE20K_473_reference.chainer -o weights/pspnet50_ADE20K_473_reference.chainer
fi
