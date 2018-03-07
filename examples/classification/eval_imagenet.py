import argparse

import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.links import FeaturePredictor
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50
from chainercv.links import VGG16

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

try:
    import cv2  # NOQA
except ImportError:
    import warnings
    warnings.warn('cv2 is not installed on your environment. '
                  'The scores reported in the ChainerCV\'s Github page '
                  'are calculated using OpenCV as the backend.'
                  'With Pillow as the backend, the scores would change.')


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument(
        '--model', choices=('vgg16', 'resnet50', 'resnet101', 'resnet152'))
    parser.add_argument('--pretrained_model', default='imagenet')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--crop', choices=('center', '10'), default='center')
    parser.add_argument('--resnet_mode', default='he')
    args = parser.parse_args()

    dataset = DirectoryParsingLabelDataset(args.val)
    label_names = directory_parsing_label_names(args.val)
    n_class = len(label_names)
    iterator = iterators.MultiprocessIterator(
        dataset, args.batchsize, repeat=False, shuffle=False,
        n_processes=6, shared_mem=300000000)

    if args.model == 'vgg16':
        extractor = VGG16(n_class, args.pretrained_model)
    elif args.model == 'resnet50':
        extractor = ResNet50(
            n_class, args.pretrained_model, arch=args.resnet_mode)
    elif args.model == 'resnet101':
        extractor = ResNet101(
            n_class, args.pretrained_model, arch=args.resnet_mode)
    elif args.model == 'resnet152':
        extractor = ResNet152(
            n_class, args.pretrained_model, arch=args.resnet_mode)
    model = FeaturePredictor(
        extractor, crop_size=224, scale_size=256, crop=args.crop)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    print('Model has been prepared. Evaluation starts.')
    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    del in_values

    pred_probs, = out_values
    gt_labels, = rest_values

    accuracy = F.accuracy(
        np.array(list(pred_probs)), np.array(list(gt_labels))).data
    print()
    print('Top 1 Error {}'.format(1. - accuracy))


if __name__ == '__main__':
    main()
