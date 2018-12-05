from __future__ import division

import argparse

import chainer

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.evaluations import eval_semantic_segmentation
from chainercv.links import SegNetBasic
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', type=str, default='camvid')
    parser.add_argument('--batchsize', type=int, default=24)
    args = parser.parse_args()

    model = SegNetBasic(
        n_class=len(camvid_label_names),
        pretrained_model=args.pretrained_model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    test = CamVidDataset(split='test')
    it = chainer.iterators.SerialIterator(test, batch_size=args.batchsize,
                                          repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, it, hook=ProgressHook(len(test)))
    # Delete an iterator of images to save memory usage.
    del in_values
    pred_labels, = out_values
    gt_labels, = rest_values

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
