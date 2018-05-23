from __future__ import division

from math import ceil
import numpy as np

import chainer
import chainer.functions as F

from chainercv.experimental.links.model.pspnet.transforms import \
    convolution_crop
from chainercv import transforms


def semantic_segmentation_predict(
        call_func, imgs, scales, mean, input_size, n_class, xp):
    labels = []
    for img in imgs:
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            if scales is not None:
                scores = _multiscale_predict(
                    call_func, img, scales, mean, input_size, n_class, xp)
            else:
                scores = _tile_predict(
                    call_func, img, mean, input_size, n_class, xp)
        labels.append(chainer.cuda.to_cpu(
            xp.argmax(scores, axis=0).astype(np.int32)))
    return labels


def _simple_predict(call_func, imgs, xp):
    xs = chainer.Variable(xp.asarray(imgs))
    with chainer.using_config('train', False):
        scores = F.softmax(call_func(xs)).array
    return scores


def _tile_predict(call_func, img, mean, input_size, n_class, xp):
    if mean is not None:
        img = img - mean
    ori_H, ori_W = img.shape[1:]
    long_size = max(ori_H, ori_W)

    if long_size > max(input_size):
        stride_rate = 2 / 3
        stride = (int(ceil(input_size[0] * stride_rate)),
                  int(ceil(input_size[1] * stride_rate)))

        imgs, param = convolution_crop(
            img, input_size, stride, return_param=True)

        counts = xp.zeros((1, ori_H, ori_W), dtype=np.float32)
        preds = xp.zeros((1, n_class, ori_H, ori_W),
                         dtype=np.float32)
        N = len(param['y_slices'])
        for i in range(N):
            img_i = imgs[i:i+1]
            y_slice = param['y_slices'][i]
            x_slice = param['x_slices'][i]
            crop_y_slice = param['crop_y_slices'][i]
            crop_x_slice = param['crop_x_slices'][i]

            scores_i = _simple_predict(call_func, img_i, xp)
            # Flip horizontally flipped score maps again
            flipped_scores_i = _simple_predict(
                call_func, img_i[:, :, :, ::-1], xp)[:, :, :, ::-1]

            preds[0, :, y_slice, x_slice] +=\
                scores_i[0, :, crop_y_slice, crop_x_slice]
            preds[0, :, y_slice, x_slice] +=\
                flipped_scores_i[0, :, crop_y_slice, crop_x_slice]
            counts[0, y_slice, x_slice] += 2

        scores = preds / counts[:, None]
    else:
        img, param = transforms.resize_contain(
            img, input_size, return_param=True)
        preds1 = _simple_predict(call_func, img[np.newaxis], xp)
        preds2 = _simple_predict(call_func, img[np.newaxis, :, :, ::-1], xp)
        preds = (preds1 + preds2[:, :, :, ::-1]) / 2

        y_start = param['y_offset']
        y_end = y_start + param['scaled_size'][0]
        x_start = param['x_offset']
        x_end = x_start + param['scaled_size'][1]
        scores = preds[:, :, y_start:y_end, x_start:x_end]
    scores = F.resize_images(scores, (ori_H, ori_W))[0].array
    return scores


def _multiscale_predict(
        call_func, img, scales, mean, input_size, n_class, xp):
    orig_H, orig_W = img.shape[1:]
    scores = []
    orig_img = img
    for scale in scales:
        img = orig_img.copy()
        if scale != 1.0:
            img = transforms.resize(
                img, (int(orig_H * scale), int(orig_W * scale)))
        # This method should return scores
        y = _tile_predict(call_func, img, mean, input_size, n_class, xp)
        assert y.shape[2:] == img.shape[1:]

        if scale != 1.0:
            y = F.resize_images(y, (orig_H, orig_W)).array
        scores.append(y)
    xp = chainer.cuda.get_array_module(scores[0])
    scores = xp.stack(scores)
    return scores.mean(0)[0]  # (C, H, W)
