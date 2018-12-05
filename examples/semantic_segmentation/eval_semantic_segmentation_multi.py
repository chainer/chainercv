from __future__ import division

import argparse
import numpy as np

import chainer
from chainer import iterators

import chainermn

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset

from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.evaluations import calc_semantic_segmentation_iou
from chainercv.experimental.links import PSPNetResNet101
from chainercv.links import SegNetBasic
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=('cityscapes', 'ade20k', 'camvid'))
    parser.add_argument(
        '--model', choices=(
            'pspnet_resnet101', 'segnet'))
    parser.add_argument('--pretrained-model')
    args = parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    if args.dataset == 'cityscapes':
        dataset = CityscapesSemanticSegmentationDataset(
            split='val', label_resolution='fine')
        label_names = cityscapes_semantic_segmentation_label_names
    elif args.dataset == 'ade20k':
        dataset = ADE20KSemanticSegmentationDataset(split='val')
        label_names = ade20k_semantic_segmentation_label_names
    elif args.dataset == 'camvid':
        dataset = CamVidDataset(split='test')
        label_names = camvid_label_names

    if args.pretrained_model:
        pretrained_model = args.pretrained_model
    else:
        pretrained_model = args.dataset
    if args.model == 'pspnet_resnet101':
        model = PSPNetResNet101(
            n_class=len(label_names),
            pretrained_model=pretrained_model, input_size=(713, 713)
        )
    elif args.model == 'segnet':
        model = SegNetBasic(
            n_class=len(label_names), pretrained_model=pretrained_model)

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    if comm.rank == 0:
        indices = np.arange(len(dataset))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm)
    dataset = dataset.slice[indices]

    it = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, it, hook=ProgressHook(len(dataset)))
    # Delete an iterator of images to save memory usage.
    del in_values
    pred_labels, = out_values
    gt_labels, = rest_values

    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    confusion = comm.allreduce(confusion)

    if comm.rank == 0:
        iou = calc_semantic_segmentation_iou(confusion)
        pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
        class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

        for iu, label_name in zip(iou, label_names):
            print('{:>23} : {:.4f}'.format(label_name, iu))
        print('=' * 34)
        print('{:>23} : {:.4f}'.format('mean IoU', np.nanmean(iou)))
        print('{:>23} : {:.4f}'.format(
            'Class average accuracy', np.nanmean(class_accuracy)))
        print('{:>23} : {:.4f}'.format(
            'Global average accuracy', pixel_accuracy))


if __name__ == '__main__':
    main()
