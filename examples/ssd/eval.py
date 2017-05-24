import argparse
import sys

import chainer
from chainer import iterators

from chainercv.datasets import VOCDetectionDataset
from chainercv.evaluations import eval_detection_voc
from chainercv.links import SSD300
from chainercv.links import SSD512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(pretrained_model='voc0712')
    elif args.model == 'ssd512':
        model = SSD512(pretrained_model='voc0712')

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    dataset = VOCDetectionDataset(
        year='2007', split='test', use_difficult=True, return_difficult=True)
    iterator = iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    pred_bboxes = list()
    pred_labels = list()
    pred_scores = list()
    gt_bboxes = list()
    gt_labels = list()
    gt_difficults = list()

    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        imgs, bboxes, labels, difficults = zip(*batch)
        gt_bboxes.extend(bboxes)
        gt_labels.extend(labels)
        gt_difficults.extend(difficults)

        bboxes, labels, scores = model.predict(imgs)
        pred_bboxes.extend(bboxes)
        pred_labels.extend(labels)
        pred_scores.extend(scores)

        sys.stdout.write('\r{:d} images'.format(len(gt_bboxes)))
        sys.stdout.flush()

    eval_ = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    print('mAP: ', eval_['map'])


if __name__ == '__main__':
    main()
