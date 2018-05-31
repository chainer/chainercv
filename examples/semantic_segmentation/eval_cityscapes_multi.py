import argparse
import numpy as np

import chainer
from chainer import iterators

import chainermn

from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.evaluations import calc_semantic_segmentation_iou
from chainercv.experimental.links import PSPNetResNet101
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('pspnet_resnet101',),
        default='pspnet_resnet101')
    parser.add_argument('--pretrained-model')
    args = parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    if args.model == 'pspnet_resnet101':
        if args.pretrained_model:
            model = PSPNetResNet101(
                n_class=len(cityscapes_semantic_segmentation_label_names),
                pretrained_model=args.pretrained_model, input_size=(713, 713)
            )
        else:
            model = PSPNetResNet101(pretrained_model='cityscapes')

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    dataset = CityscapesSemanticSegmentationDataset(
        split='val', label_resolution='fine')

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

        for iu, label_name in zip(
                iou, cityscapes_semantic_segmentation_label_names):
            print('{:>23} : {:.4f}'.format(label_name, iu))
        print('=' * 34)
        print('{:>23} : {:.4f}'.format('mean IoU', np.nanmean(iou)))
        print('{:>23} : {:.4f}'.format(
            'Class average accuracy', np.nanmean(class_accuracy)))
        print('{:>23} : {:.4f}'.format(
            'Global average accuracy', np.nanmean(pixel_accuracy)))


if __name__ == '__main__':
    main()
