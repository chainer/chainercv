import argparse
import numpy as np

import chainer
from chainer import iterators

import chainermn

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.evaluations import calc_semantic_segmentation_iou
from chainercv.experimental.links import PSPNetResNet101
from chainercv.experimental.links import PSPNetResNet50
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('pspnet_resnet50', 'pspnet_resnet101'),
        default='pspnet_resnet101')
    parser.add_argument('--pretrained-model')
    args = parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    n_class = len(ade20k_semantic_segmentation_label_names)
    if args.pretrained_model:
        pretrained_model = args.pretrained_model
    else:
        pretrained_model = 'ade20k'
    if args.model == 'pspnet_resnet50':
        model = PSPNetResNet50(
            n_class=n_class, pretrained_model=pretrained_model,
            input_size=(473, 473))
    elif args.model == 'pspnet_resnet101':
        model = PSPNetResNet101(
            n_class=n_class, pretrained_model=pretrained_model,
            input_size=(473, 473))

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    dataset = ADE20KSemanticSegmentationDataset(split='val')

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
                iou, ade20k_semantic_segmentation_label_names):
            print('{:>23} : {:.4f}'.format(label_name, iu))
        print('=' * 34)
        print('{:>23} : {:.4f}'.format('mean IoU', np.nanmean(iou)))
        print('{:>23} : {:.4f}'.format(
            'Class average accuracy', np.nanmean(class_accuracy)))
        print('{:>23} : {:.4f}'.format(
            'Global average accuracy', pixel_accuracy))


if __name__ == '__main__':
    main()
