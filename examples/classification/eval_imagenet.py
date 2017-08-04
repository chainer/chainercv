import argparse
import sys
import time

import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingClassificationDataset
from chainercv.links import FeatureExtractionPredictor
from chainercv.links import VGG16

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
    chainer.config.train = False

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument(
        '--model', choices=('vgg16'))
    parser.add_argument('--pretrained_model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--crop', type=str, default='center')
    args = parser.parse_args()

    dataset = DirectoryParsingClassificationDataset(args.val)
    label_names = directory_parsing_label_names(args.val)
    iterator = iterators.MultiprocessIterator(
        dataset, args.batchsize, repeat=False, shuffle=False,
        n_processes=6, shared_mem=300000000)

    if args.model == 'vgg16':
        if args.pretrained_model:
            extractor = VGG16(pretrained_model=args.pretrained_model,
                              n_class=len(label_names))
        else:
            extractor = VGG16(pretrained_model='imagenet',
                              n_class=len(label_names))
    model = FeatureExtractionPredictor(extractor, crop=args.crop)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    print('Model has been prepared. Evaluation starts.')
    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    del imgs

    pred_scores, = pred_values
    gt_scores, = gt_values

    accuracy = F.accuracy(
        np.array(list(pred_scores)), np.array(list(gt_scores))).data
    print()
    print('Top 1 Error {}'.format(1. - accuracy))


if __name__ == '__main__':
    main()
