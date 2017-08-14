#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA  # isort:skip
sys.path.insert(0, '.')  # NOQA  # isort:skip

import argparse
import os
from math import ceil

import chainer
import chainer.functions as F
import numpy as np
from chainer import serializers
from chainer.datasets import TransformDataset
from chainercv import transforms
from skimage import io

from datasets import cityscapes_labels
from datasets import CityscapesSemanticSegmentationDataset
from datasets import VOCSemanticSegmentationDataset
from pspnet import PSPNet
from tqdm import tqdm


def preprocess(inputs):
    if not isinstance(inputs, np.ndarray) and len(inputs) == 2:
        img, label = inputs
    elif isinstance(inputs, np.ndarray):
        img = inputs
    else:
        raise ValueError('{}'.format(inputs))

    img = img[::-1]  # Convert to BGR order

    # Mean values in BGR order
    mean = np.array([103.939, 116.779, 123.68])

    # Mean subtraction
    img -= mean[:, None, None]

    if not isinstance(inputs, np.ndarray) and len(inputs) == 2:
        return img, label
    elif isinstance(inputs, np.ndarray):
        return img
    else:
        raise ValueError('{}'.format(inputs))


def predict(model, img):
    if img.ndim == 3:
        img = img[None, ...]
    imgs = np.concatenate([img, img[:, :, :, ::-1]], axis=0)
    scores = model.predict(imgs, argmax=False)
    score = (scores[0] + scores[1][:, :, ::-1])[None, ...]
    return F.softmax(score).data


def pad_img(img, crop_size):
    if img.shape[1] < crop_size:
        pad_h = crop_size - img.shape[1]
        img = np.pad(img, ((0, 0), (0, pad_h), (0, 0)), 'constant')
    else:
        pad_h = 0
    if img.shape[2] < crop_size:
        pad_w = crop_size - img.shape[2]
        img = np.pad(img, ((0, 0), (0, 0), (0, pad_w)), 'constant')
    else:
        pad_w = 0
    assert img.shape[1:] == (crop_size, crop_size)
    return img, pad_h, pad_w


def scale_process(model, img, n_class, base_size, crop_size, scale):
    ori_rows, ori_cols = img.shape[1:]
    long_size = int(base_size * scale)
    new_rows, new_cols = long_size, long_size
    if ori_rows > ori_cols:
        new_cols = int(long_size / ori_rows * ori_cols)
    else:
        new_rows = int(long_size / ori_cols * ori_rows)
    if ori_rows != new_rows and ori_cols != new_cols:
        img_scaled = transforms.resize(img, (new_rows, new_cols))
    else:
        img_scaled = img
    long_size = max(new_rows, new_cols)
    if long_size > crop_size:
        count = np.zeros((new_rows, new_cols))
        pred = np.zeros((1, n_class, new_rows, new_cols))
        stride_rate = chainer.config.stride_rate
        stride = ceil(crop_size * stride_rate)
        hh = ceil((new_rows - crop_size) / stride) + 1
        ww = ceil((new_cols - crop_size) / stride) + 1
        for yy in range(hh):
            for xx in range(ww):
                sy = yy * stride
                ey = sy + crop_size
                sx = xx * stride
                ex = sx + crop_size
                img_sub = img_scaled[:, sy:ey, sx:ex]
                img_sub, pad_h, pad_w = pad_img(img_sub, crop_size)
                pred_sub = predict(model, img_sub)
                if sy + crop_size > new_rows:
                    pred_sub = pred_sub[:, :, :-pad_h, :]
                if sx + crop_size > new_cols:
                    pred_sub = pred_sub[:, :, :, :-pad_w]
                pred[:, :, sy:ey, sx:ex] = pred_sub
                count[sy:ey, sx:ex] += 1
        assert np.sum(count == 0) == 0, '{}'.format(np.sum(count == 0))
        score = (pred / count[None, None, ...]).astype(np.float32)
    else:
        img_scaled, pad_h, pad_w = pad_img(img_scaled, crop_size)
        pred = predict(model, img_scaled)
        score = pred[:, :, :crop_size - pad_h, :crop_size - pad_w]
    score = F.resize_images(score, (ori_rows, ori_cols))[0].data
    score = score / score.sum(axis=0)
    assert score.shape[1:] == (ori_rows, ori_cols), '{}'.format(score.shape)

    if chainer.config.save_test_image:
        test_pred = np.argmax(score, axis=0)
        test_out = np.zeros((ori_rows, ori_cols, 3), dtype=np.uint8)
        for label in cityscapes_labels:
            test_out[np.where(test_pred == label.trainId)] = label.color
        io.imsave('scale_{}.png'.format(scale), test_out)

    return score


def inference(model, n_class, base_size, crop_size, img, scales):
    pred = np.zeros((n_class, img.shape[1], img.shape[2]))
    if scales is not None and isinstance(scales, (list, tuple)):
        for i, scale in enumerate(scales):
            pred += scale_process(
                model, img, n_class, base_size, crop_size, scale)
        pred = pred / float(len(scales))
    else:
        pred = scale_process(model, img, n_class, base_size, crop_size, 1.0)
    pred = np.argmax(pred, axis=0)
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--scales', type=float, nargs='*', default=None)
    parser.add_argument(
        '--model', type=str, choices=['VOC', 'Cityscapes', 'ADE20K'])
    parser.add_argument('--cityscapes_img_dir', type=str, default=None)
    parser.add_argument('--voc_data_dirr', type=str, default=None)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--color_out_dir', type=str, default=None)
    parser.add_argument('--start_i', type=int)
    parser.add_argument('--end_i', type=int)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--stride_rate', type=float, default=2 / 3)
    parser.add_argument('--save_test_image', action='store_true', default=False)
    args = parser.parse_args()

    chainer.config.stride_rate = args.stride_rate
    chainer.config.save_test_image = args.save_test_image

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.color_out_dir is not None:
        if not os.path.exists(args.color_out_dir):
            os.mkdir(args.color_out_dir)

    if args.model == 'VOC':
        n_class = 21
        n_blocks = [3, 4, 23, 3]
        feat_size = 60
        mid_stride = True
        param_fn = 'weights/pspnet101_VOC2012_473_reference.chainer'
        base_size = 512
        crop_size = 473
        dataset = VOCSemanticSegmentationDataset(
            args.voc_data_dir, split='test')
    elif args.model == 'Cityscapes':
        n_class = 19
        n_blocks = [3, 4, 23, 3]
        feat_size = 90
        mid_stride = True
        param_fn = 'weights/pspnet101_cityscapes_713_reference.chainer'
        base_size = 2048
        crop_size = 713
        dataset = CityscapesSemanticSegmentationDataset(
            args.cityscapes_img_dir, None, args.split)
    elif args.model == 'ADE20K':
        n_class = 150
        n_blocks = [3, 4, 6, 3]
        feat_size = 60
        mid_stride = False
        param_fn = 'weights/pspnet101_ADE20K_473_reference.chainer'
        base_size = 512
        crop_size = 473

    dataset = TransformDataset(dataset, preprocess)
    print('dataset:', len(dataset))

    chainer.config.train = False
    model = PSPNet(n_class, n_blocks, feat_size, mid_stride=mid_stride)
    serializers.load_npz(param_fn, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu)
        model.to_gpu(args.gpu)

    for i in tqdm(range(args.start_i, args.end_i + 1)):
        img = dataset[i]
        out_fn = os.path.join(
            args.out_dir, os.path.basename(dataset._dataset.img_fns[i]))
        pred = inference(
            model, n_class, base_size, crop_size, img, args.scales)
        assert pred.ndim == 2

        if args.model == 'Cityscapes':
            if args.color_out_dir is not None:
                color_out = np.zeros(
                    (pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            label_out = np.zeros_like(pred)
            for label in cityscapes_labels:
                label_out[np.where(pred == label.trainId)] = label.id
                if args.color_out_dir is not None:
                    color_out[np.where(pred == label.trainId)] = label.color
            pred = label_out

            if args.color_out_dir is not None:
                base_fn = os.path.basename(dataset._dataset.img_fns[i])
                base_fn = os.path.splitext(base_fn)[0]
                color_fn = os.path.join(args.color_out_dir, base_fn)
                color_fn += '_color.png'
                io.imsave(color_fn, color_out)
        io.imsave(out_fn, pred)
