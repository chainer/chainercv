from __future__ import division

import argparse

import chainer
from chainer import cuda
from chainer.dataset import concat_examples
import chainer.functions as F
from chainer import serializers

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.links import SegNetBasic

import numpy as np


def calc_bn_statistics(model, gpu):
    model.to_gpu(gpu)

    d = CamVidDataset(split='train')
    it = chainer.iterators.SerialIterator(d, 24, repeat=False, shuffle=False)
    bn_params = {}
    num_iterations = 0
    for batch in it:
        imgs, labels = concat_examples(batch, device=gpu)
        model(imgs)
        for name, link in model.namedlinks():
            if name.endswith('_bn'):
                if name not in bn_params:
                    bn_params[name] = [cuda.to_cpu(link.avg_mean),
                                       cuda.to_cpu(link.avg_var)]
                else:
                    bn_params[name][0] += cuda.to_cpu(link.avg_mean)
                    bn_params[name][1] += cuda.to_cpu(link.avg_var)
        num_iterations += 1

    for name, params in bn_params.items():
        bn_params[name][0] /= num_iterations
        bn_params[name][1] /= num_iterations

    for name, link in model.namedlinks():
        if name.endswith('_bn'):
            link.avg_mean = bn_params[name][0]
            link.avg_var = bn_params[name][1]

    model.to_cpu()
    return model


def main():
    # This follows evaluation code used in SegNet.
    # https://github.com/alexgkendall/SegNet-Tutorial/blob/master/
    # # Scripts/compute_test_results.m
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int, default=-1)
    parser.add_argument('snapshot', type=str)
    parser.add_argument('--batchsize', type=int, default=24)
    args = parser.parse_args()

    n_class = 11
    ignore_labels = [-1]

    model = SegNetBasic(n_class=n_class)
    serializers.load_npz(args.snapshot, model)
    model = calc_bn_statistics(model, args.gpu)
    model.train = False
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test = CamVidDataset(split='test')
    it = chainer.iterators.SerialIterator(test, batch_size=args.batchsize,
                                          repeat=False, shuffle=False)

    n_positive = [0 for _ in range(n_class)]
    n_true = [0 for _ in range(n_class)]
    n_true_positive = [0 for _ in range(n_class)]
    for batch in it:
        img, gt_label = concat_examples(batch, args.gpu)
        img = chainer.Variable(img, volatile=True)
        pred_label = F.argmax(F.softmax(model(img)), axis=1)
        pred_label = cuda.to_cpu(pred_label.data)
        gt_label = cuda.to_cpu(gt_label)
        for cls_i in range(n_class):
            if cls_i in ignore_labels:
                continue
            n_positive[cls_i] += np.sum(pred_label == cls_i)
            n_true[cls_i] += np.sum(gt_label == cls_i)
            n_true_positive[cls_i] += np.sum(
                (pred_label == cls_i) * (gt_label == cls_i))

    ious = []
    mean_accs = []
    pixel_accs = []
    for cls_i in range(n_class):
        if cls_i in ignore_labels:
            continue
        deno = n_positive[cls_i] + n_true[cls_i] - n_true_positive[cls_i]
        iou = n_true_positive[cls_i] / deno
        ious.append(iou)
        print('{:>23} : {:.4f}'.format(camvid_label_names[cls_i], iou))

        mean_accs.append(n_true_positive[cls_i] / n_true[cls_i])
        pixel_accs.append([n_true_positive[cls_i], n_true[cls_i]])

    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', np.mean(ious)))
    print('{:>23} : {:.4f}'.format(
        'Class average accuracy', np.mean(mean_accs)))
    pixel_accs = np.asarray(pixel_accs)
    pixel_accs = pixel_accs[:, 0].sum() / pixel_accs[:, 1].sum()
    print('{:>23} : {:.4f}'.format(
        'Global average accuracy', pixel_accs))


if __name__ == '__main__':
    main()
