from __future__ import division

import argparse
import sys
import time

import chainer
from chainer import iterators

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset
from chainercv.evaluations import eval_detection_voc
from chainercv.links import FasterRCNNVGG16
from chainercv.utils import apply_detection_link


class ProgressHook(object):

    def __init__(self, n_total):
        self.n_total = n_total
        self.start = time.time()
        self.n_processed = 0

    def __call__(self, pred_bboxes, pred_labels, pred_scores, gt_values):
        self.n_processed += len(pred_bboxes)
        fps = self.n_processed / (time.time() - self.start)
        sys.stdout.write(
            '\r{:d} of {:d} images, {:.2f} FPS'.format(
                self.n_processed, self.n_total, fps))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    model = FasterRCNNVGG16(pretrained_model='voc07')

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    dataset = VOCDetectionDataset(
        year='2007', split='test', use_difficult=True, return_difficult=True)
    iterator = iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    pred_bboxes, pred_labels, pred_scores, gt_values = \
        apply_detection_link(model, iterator, hook=ProgressHook(len(dataset)))
    gt_bboxes, gt_labels, gt_difficults = gt_values

    eval_ = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    print()
    print('mAP: {:f}'.format(eval_['map']))
    for l, name in enumerate(voc_detection_label_names):
        if l in eval_:
            print('{:s}: {:f}'.format(name, eval_[l]['ap']))
        else:
            print('{:s}: -'.format(name))


if __name__ == '__main__':
    main()
