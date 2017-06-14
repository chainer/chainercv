import argparse
import random

import numpy as np

import chainer
import chainer.links as L
from chainer import iterators
from chainer import training
from chainer.training import extensions

from chainercv.datasets import ImageFolderDataset
from chainercv.links import VGG16Layers


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    dataset = ImageFolderDataset(args.val)
    iterator = iterators.MultiprocessIterator(
        dataset, args.batchsize, repeat=False, shuffle=False,
        n_processes=4)

    model = L.Classifier(VGG16Layers())

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    
    result = extensions.Evaluator(iterator, model)
    print result


if __name__ == '__main__':
    main()
