import argparse

import chainer
from chainer import iterators

import chainermn

from chainercv.evaluations import eval_semantic_segmentation
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from eval_semantic_segmentation import get_dataset_and_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=('cityscapes', 'ade20k', 'camvid', 'voc'))
    parser.add_argument(
        '--model', choices=(
            'pspnet_resnet101', 'segnet', 'deeplab_v3plus_xception65'))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--input-size', type=int, default=None)
    args = parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    if args.input_size is None:
        input_size = None
    else:
        input_size = (args.input_size, args.input_size)

    dataset, label_names, model = get_dataset_and_model(
        args.dataset, args.model, args.pretrained_model,
        input_size)

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    if not comm.rank == 0:
        apply_to_iterator(model.predict, None, comm=comm)
        return

    it = iterators.MultithreadIterator(
        dataset, comm.size, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, it, hook=ProgressHook(len(dataset)), comm=comm)
    # Delete an iterator of images to save memory usage.
    del in_values
    pred_labels, = out_values
    gt_labels, = rest_values

    result = eval_semantic_segmentation(pred_labels, gt_labels)

    for iu, label_name in zip(result['iou'], label_names):
        print('{:>23} : {:.4f}'.format(label_name, iu))
    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', result['miou']))
    print('{:>23} : {:.4f}'.format(
        'Class average accuracy', result['mean_class_accuracy']))
    print('{:>23} : {:.4f}'.format(
        'Global average accuracy', result['pixel_accuracy']))


if __name__ == '__main__':
    main()
