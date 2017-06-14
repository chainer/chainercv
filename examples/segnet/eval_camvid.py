from __future__ import division

import argparse
from collections import defaultdict
import numpy as np

import chainer
from chainer.dataset import concat_examples

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.evaluations import eval_semantic_segmentation
from chainercv.links import SegNetBasic
from chainercv.utils import apply_prediction_to_iterator


def calc_bn_statistics(model, batchsize):
    train = CamVidDataset(split='train')
    it = chainer.iterators.SerialIterator(
        train, batchsize, repeat=False, shuffle=False)
    bn_avg_mean = defaultdict(np.float32)
    bn_avg_var = defaultdict(np.float32)

    n_iter = 0
    for batch in it:
        imgs, _ = concat_examples(batch)
        model(model.xp.array(imgs))
        for name, link in model.namedlinks():
            if name.endswith('_bn'):
                bn_avg_mean[name] += link.avg_mean
                bn_avg_var[name] += link.avg_var
        n_iter += 1

    for name, link in model.namedlinks():
        if name.endswith('_bn'):
            link.avg_mean = bn_avg_mean[name] / n_iter
            link.avg_var = bn_avg_var[name] / n_iter

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', type=str, default='camvid')
    parser.add_argument('--batchsize', type=int, default=24)
    args = parser.parse_args()

    model = SegNetBasic(
        n_class=len(camvid_label_names),
        pretrained_model=args.pretrained_model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    model = calc_bn_statistics(model, args.batchsize)

    chainer.config.train = False

    test = CamVidDataset(split='test')
    it = chainer.iterators.SerialIterator(test, batch_size=args.batchsize,
                                          repeat=False, shuffle=False)

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, it)
    # Delete an iterator of images to save memory usage.
    del imgs
    pred_labels, = pred_values
    gt_labels, = gt_values

    result = eval_semantic_segmentation(pred_labels, gt_labels)

    for iu, label_name in zip(result['iou'], camvid_label_names):
        print('{:>23} : {:.4f}'.format(label_name, iu))
    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', result['miou']))
    print('{:>23} : {:.4f}'.format(
        'Class average accuracy', result['mean_class_accuracy']))
    print('{:>23} : {:.4f}'.format(
        'Global average accuracy', result['pixel_accuracy']))


if __name__ == '__main__':
    main()
