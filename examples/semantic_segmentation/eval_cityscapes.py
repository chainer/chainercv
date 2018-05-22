import argparse

import chainer
from chainer import iterators

from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.evaluations import eval_semantic_segmentation
from chainercv.experimental.links import PSPNetResNet101
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('pspnet_resnet101'),
        default='pspnet_resnet101')
    parser.add_argument('--pretrained_model')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    if args.model == 'pspnet_resnet101':
        if args.pretrained_model:
            model = PSPNetResNet101(
                n_class=len(cityscapes_semantic_segmentation_label_names),
                pretrained_model=args.pretrained_model, input_size=(713, 713)
            )
        else:
            model = PSPNetResNet101(pretrained_model='cityscapes')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = CityscapesSemanticSegmentationDataset(
        split='val', label_resolution='fine')
    it = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, it, hook=ProgressHook(len(dataset)))
    # Delete an iterator of images to save memory usage.
    del in_values
    pred_labels, = out_values
    gt_labels, = rest_values

    result = eval_semantic_segmentation(pred_labels, gt_labels)

    for iu, label_name in zip(
            result['iou'], cityscapes_semantic_segmentation_label_names):
        print('{:>23} : {:.4f}'.format(label_name, iu))
    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', result['miou']))
    print('{:>23} : {:.4f}'.format(
        'Class average accuracy', result['mean_class_accuracy']))
    print('{:>23} : {:.4f}'.format(
        'Global average accuracy', result['pixel_accuracy']))


if __name__ == '__main__':
    main()
