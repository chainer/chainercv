import argparse

import chainer
from chainer import iterators

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.datasets import VOCSemanticSegmentationDataset

from chainercv.evaluations import eval_semantic_segmentation
from chainercv.experimental.links import PSPNetResNet101
from chainercv.experimental.links import PSPNetResNet50
from chainercv.links import DeepLabV3plusXception65
from chainercv.links import SegNetBasic
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


def get_dataset_and_model(dataset_name, model_name, pretrained_model,
                          input_size):
    if dataset_name == 'cityscapes':
        dataset = CityscapesSemanticSegmentationDataset(
            split='val', label_resolution='fine')
        label_names = cityscapes_semantic_segmentation_label_names
    elif dataset_name == 'ade20k':
        dataset = ADE20KSemanticSegmentationDataset(split='val')
        label_names = ade20k_semantic_segmentation_label_names
    elif dataset_name == 'camvid':
        dataset = CamVidDataset(split='test')
        label_names = camvid_label_names
    elif dataset_name == 'voc':
        dataset = VOCSemanticSegmentationDataset(split='val')
        label_names = voc_semantic_segmentation_label_names

    n_class = len(label_names)

    if pretrained_model:
        pretrained_model = pretrained_model
    else:
        pretrained_model = dataset_name
    if model_name == 'pspnet_resnet101':
        model = PSPNetResNet101(
            n_class=n_class,
            pretrained_model=pretrained_model,
            input_size=input_size
        )
    elif model_name == 'pspnet_resnet50':
        model = PSPNetResNet50(
            n_class=n_class,
            pretrained_model=pretrained_model,
            input_size=input_size
        )
    elif model_name == 'segnet':
        model = SegNetBasic(
            n_class=n_class, pretrained_model=pretrained_model)
    elif model_name == 'deeplab_v3plus_xception65':
        model = DeepLabV3plusXception65(
            n_class=n_class,
            pretrained_model=pretrained_model,
            min_input_size=input_size)

    return dataset, label_names, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=('cityscapes', 'ade20k', 'camvid', 'voc'))
    parser.add_argument(
        '--model', choices=(
            'pspnet_resnet101', 'segnet', 'deeplab_v3plus_xception65'))
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    parser.add_argument('--input-size', type=int, default=None)
    args = parser.parse_args()

    if args.input_size is None:
        input_size = None
    else:
        input_size = (args.input_size, args.input_size)

    dataset, label_names, model = get_dataset_and_model(
        args.dataset, args.model, args.pretrained_model,
        input_size)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    it = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, it, hook=ProgressHook(len(dataset)))
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
