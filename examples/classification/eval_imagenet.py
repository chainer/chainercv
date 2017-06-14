import argparse
import random
import sys
import time

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


class ProgressHook(object):

    def __init__(self, n_total):
        self.n_total = n_total
        self.start = time.time()
        self.n_processed = 0

    def __call__(self, imgs, pred_values, gt_values):
        self.n_processed += len(imgs)
        fps = self.n_processed / (time.time() - self.start)
        sys.stdout.write(
            '\r{:d} of {:d} images, {:.2f} FPS'.format(
                self.n_processed, self.n_total, fps))
        sys.stdout.flush()


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
        n_processes=4, shared_mem=10000000)

    model = VGG16Layers(pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    del imgs

    pred_labels, = pred_values
    gt_labels, = gt_values

    accuracy = F.accuracy(
        np.array(list(pred_labels)), np.array(list(gt_labels))).data
    print()
    print('Top 1 Error {}'.format(1. - accuracy))


if __name__ == '__main__':
    main()
