from __future__ import division

import argparse
import numpy as np

import chainer
from chainer import cuda
from chainer.dataset import concat_examples

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.evaluations import calc_semantic_segmentation_iou
from chainercv.links import SegNetBasic
from chainercv.utils import apply_prediction_to_iterator


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', type=str, default='camvid')
    parser.add_argument('--batchsize', type=int, default=24)
    args = parser.parse_args()

    n_class = 11

    model = SegNetBasic(
        n_class=n_class, pretrained_model=args.pretrained_model)
    model = calc_bn_statistics(model, args.gpu)
    chainer.config.train = False
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test = CamVidDataset(split='test')
    it = chainer.iterators.SerialIterator(test, batch_size=args.batchsize,
                                          repeat=False, shuffle=False)

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, it)
    # Delete an iterator of images to save memory usage.
    del imgs
    pred_labels, = pred_values
    gt_labels, = gt_values

    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    ious = calc_semantic_segmentation_iou(confusion)

    pixel_accuracy = np.diag(confusion).sum() / confusion.size
    mean_pixel_accuracy = np.mean(
        np.diag(confusion) / np.sum(confusion, axis=1))

    for iou, label_name in zip(ious, camvid_label_names):
        print('{:>23} : {:.4f}'.format(label_name, iou))
    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', np.nanmean(ious)))
    print('{:>23} : {:.4f}'.format(
        'Class average accuracy', mean_pixel_accuracy))
    print('{:>23} : {:.4f}'.format(
        'Global average accuracy', pixel_accuracy))


if __name__ == '__main__':
    main()
