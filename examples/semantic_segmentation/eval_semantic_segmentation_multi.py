from __future__ import division

import argparse
import multiprocessing
import numpy as np

import chainer
from chainer import iterators

import chainermn

from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.evaluations import calc_semantic_segmentation_iou
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from eval_semantic_segmentation import get_dataset_and_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=('cityscapes', 'ade20k', 'camvid'))
    parser.add_argument(
        '--model', choices=(
            'pspnet_resnet101', 'segnet'))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--input-size', type=int, default=None)
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    if args.input_size is None:
        input_size = None
    else:
        input_size = (args.input_size, args.input_size)

    dataset, label_names, model = get_dataset_and_model(
        args.dataset, args.model, args.pretrained_model,
        input_size)
    assert len(dataset) % comm.size == 0, \
        "The size of the dataset should be a multiple "\
        "of the number of GPUs"

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
