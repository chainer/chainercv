import argparse
import random

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators
from chainer import training
from chainer.training import extensions

from chainercv.datasets import ImageFolderDataset
from chainercv.links import VGG16Layers

from chainercv.utils import apply_prediction_to_iterator


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--pretrained_model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    dataset = ImageFolderDataset(args.val)
    iterator = iterators.MultiprocessIterator(
        dataset, args.batchsize, repeat=False, shuffle=False,
        n_processes=4)

    model = VGG16Layers(pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    imgs, pred_values, gt_values = apply_prediction_to_iterator(model.predict, iterator)
    del imgs

    pred_labels, = pred_values
    gt_labels, = gt_values

    accuracy = F.accuracy(
        np.array(list(pred_labels)), np.array(list(gt_labels))).data
    print accuracy


if __name__ == '__main__':
    main()
