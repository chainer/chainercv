from __future__ import division

import argparse

import chainer
from chainer import cuda
from chainer.dataset import concat_examples
import chainer.functions as F
from chainer import serializers
from chainercv.datasets import CamVidDataset
from chainercv.datasets.camvid.camvid_dataset import camvid_label_names
from chainercv.links import SegNetBasic

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--batchsize', type=int, default=24)
    parser.add_argument('--n_class', type=int, default=11)
    parser.add_argument('--ignore_labels', type=int, nargs='*', default=[11])
    args = parser.parse_args()

    model = SegNetBasic(args.n_class)
    model.train = False
    serializers.load_npz(args.snapshot, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test = CamVidDataset(split='test')

    n_positive = [0 for _ in range(args.n_class)]
    n_true = [0 for _ in range(args.n_class)]
    n_true_positive = [0 for _ in range(args.n_class)]
    for i in range(0, len(test), args.batchsize):
        img, label = concat_examples(test[i:i + args.batchsize], args.gpu)
        img = chainer.Variable(img, volatile=True)
        y = F.argmax(F.softmax(model(img)), axis=1)
        y, t = cuda.to_cpu(y.data), cuda.to_cpu(label)
        for cls_i in range(args.n_class):
            if cls_i in args.ignore_labels:
                continue
            n_positive[cls_i] += np.sum(y == cls_i)
            n_true[cls_i] += np.sum(t == cls_i)
            n_true_positive[cls_i] += np.sum((y == cls_i) * (t == cls_i))

    ious = []
    class_ave_acc = []
    global_ave_acc = []
    for cls_i in range(args.n_class):
        if cls_i in args.ignore_labels:
            continue
        deno = n_positive[cls_i] + n_true[cls_i] - n_true_positive[cls_i]
        iou = n_true_positive[cls_i] / deno
        ious.append(iou)
        print('{:>23} : {:.4f}'.format(camvid_label_names[cls_i], iou))

        class_ave_acc.append(n_true_positive[cls_i] / n_true[cls_i])
        global_ave_acc.append([n_true_positive[cls_i], n_true[cls_i]])

    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', np.mean(ious)))
    print('{:>23} : {:.4f}'.format(
            'Class average accuracy', np.mean(class_ave_acc)))
    global_ave_acc = np.asarray(global_ave_acc)
    global_ave_acc = global_ave_acc[:, 0].sum() / global_ave_acc[:, 1].sum()
    print('{:>23} : {:.4f}'.format(
            'Global average accuracy', global_ave_acc))
