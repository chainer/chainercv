import argparse

import copy
import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.links import FeaturePredictor
from chainercv.links import MobileNetV2
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50
from chainercv.links import SEResNet101
from chainercv.links import SEResNet152
from chainercv.links import SEResNet50
from chainercv.links import SEResNeXt101
from chainercv.links import SEResNeXt50
from chainercv.links import VGG16

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


models = {
    # model: (class, dataset -> pretrained_model, default batchsize,
    #         crop, resnet_arch)
    'vgg16': (VGG16, {}, 32, 'center', None),
    'resnet50': (ResNet50, {}, 32, 'center', 'fb'),
    'resnet101': (ResNet101, {}, 32, 'center', 'fb'),
    'resnet152': (ResNet152, {}, 32, 'center', 'fb'),
    'se-resnet50': (SEResNet50, {}, 32, 'center', None),
    'se-resnet101': (SEResNet101, {}, 32, 'center', None),
    'se-resnet152': (SEResNet152, {}, 32, 'center', None),
    'se-resnext50': (SEResNeXt50, {}, 32, 'center', None),
    'se-resnext101': (SEResNeXt101, {}, 32, 'center', None),
    'mobilenet_v2_1.0': (MobileNetV2, {}, 32, 'center', None),
    'mobilenet_v2_1.4': (MobileNetV2, {}, 32, 'center', None)
}


def setup(dataset, model, pretrained_model, batchsize, val, crop, resnet_arch):
    dataset_name = dataset
    if dataset_name == 'imagenet':
        dataset = DirectoryParsingLabelDataset(val)
        label_names = directory_parsing_label_names(val)

    def eval_(out_values, rest_values):
        pred_probs, = out_values
        gt_labels, = rest_values

        accuracy = F.accuracy(
            np.array(list(pred_probs)), np.array(list(gt_labels))).data
        print()
        print('Top 1 Error {}'.format(1. - accuracy))

    cls, pretrained_models, default_batchsize = models[model][:3]
    if pretrained_model is None:
        pretrained_model = pretrained_models.get(dataset_name, dataset_name)
    if crop is None:
        crop = models[model][3]
    kwargs = {'pretrained_model': pretrained_model}
    if model in ['resnet50', 'resnet101', 'resnet152']:
        if resnet_arch is None:
            resnet_arch = models[model][4]
        kwargs['arch'] = resnet_arch
    params = copy.deepcopy(cls.preset_params[dataset_name])
    params['n_class'] = len(label_names)
    kwargs.update(params)
    extractor = cls(**kwargs)
    model = FeaturePredictor(
        extractor, crop_size=224, scale_size=256, crop=crop)

    if batchsize is None:
        batchsize = default_batchsize

    return dataset, eval_, model, batchsize


def main():
    parser = argparse.ArgumentParser(
        description='Evaluating convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--model', choices=sorted(models.keys()))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--dataset', choices=('imagenet',))
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--crop', choices=('center', '10'))
    parser.add_argument('--resnet-arch')
    args = parser.parse_args()

    dataset, eval_, model, batchsize = setup(
        args.dataset, args.model, args.pretrained_model, args.batchsize,
        args.val, args.crop, args.resnet_arch)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    iterator = iterators.MultiprocessIterator(
        dataset, batchsize, repeat=False, shuffle=False,
        n_processes=6, shared_mem=300000000)

    print('Model has been prepared. Evaluation starts.')
    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    del in_values

    eval_(out_values, rest_values)


if __name__ == '__main__':
    main()
