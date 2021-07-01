import argparse

import numpy as np
import os

import chainer
from chainer.links.caffe.caffe_function import CaffeFunction

from chainercv.links import SEResNet101
from chainercv.links import SEResNet152
from chainercv.links import SEResNet50
from chainercv.links import SEResNeXt101
from chainercv.links import SEResNeXt50


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'conv{}_{}'.format(bname, cname))
    src_bn = getattr(src, 'conv{}_{}/bn'.format(bname, cname))
    src_scale = getattr(src, 'conv{}_{}/bn/scale'.format(bname, cname))
    if dst_conv.groups == 1:
        dst_conv.W.data[:] = src_conv.W.data
    else:
        group_size = src_conv.W.data.shape[1] // dst_conv.groups
        for group in range(dst_conv.groups):
            from_idx = group_size * group
            to_idx = group_size * (group + 1)
            dst_conv.W.data[from_idx: to_idx, :, :, :] = \
                src_conv.W.data[from_idx: to_idx, from_idx: to_idx, :, :]
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_se_components(src, dst_se, bname, cname):
    src_se_down = getattr(src, 'conv{}_{}_down'.format(bname, cname))
    src_se_up = getattr(src, 'conv{}_{}_up'.format(bname, cname))
    hidden_size, in_size = dst_se.down.W.shape
    dst_se.down.W.data[:] = src_se_down.W.data.reshape((hidden_size, in_size))
    dst_se.down.b.data[:] = src_se_down.b.data
    dst_se.up.W.data[:] = src_se_up.W.data.reshape((in_size, hidden_size))
    dst_se.up.b.data[:] = src_se_up.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(
        src, dst.conv1.conv, dst.conv1.bn, name, '1x1_reduce')
    _transfer_components(
        src, dst.conv2.conv, dst.conv2.bn, name, '3x3')
    _transfer_components(
        src, dst.conv3.conv, dst.conv3.bn, name, '1x1_increase')
    _transfer_components(
        src, dst.residual_conv.conv, dst.residual_conv.bn, name, '1x1_proj')
    _transfer_se_components(src, dst.se, name, '1x1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(
        src, dst.conv1.conv, dst.conv1.bn, name, '1x1_reduce')
    _transfer_components(
        src, dst.conv2.conv, dst.conv2.bn, name, '3x3')
    _transfer_components(
        src, dst.conv3.conv, dst.conv3.bn, name, '1x1_increase')
    _transfer_se_components(src, dst.se, name, '1x1')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet50(src, dst, class_indices):
    # Reorder weights to work on RGB and not on BGR
    dst.conv1.conv.W.data[:] = src['conv1/7x7_s2'].W.data[:, ::-1]
    # No bias setting for conv1, which is different from ResNet50.
    dst.conv1.bn.avg_mean[:] = src['conv1/7x7_s2/bn'].avg_mean
    dst.conv1.bn.avg_var[:] = src['conv1/7x7_s2/bn'].avg_var
    dst.conv1.bn.gamma.data[:] = src['conv1/7x7_s2/bn/scale'].W.data
    dst.conv1.bn.beta.data[:] = src['conv1/7x7_s2/bn/scale'].bias.b.data

    _transfer_block(src, dst.res2, ['2_1', '2_2', '2_3'])
    _transfer_block(src, dst.res3, ['3_1', '3_2', '3_3', '3_4'])
    _transfer_block(src, dst.res4, ['4_1', '4_2', '4_3', '4_4', '4_5', '4_6'])
    _transfer_block(src, dst.res5, ['5_1', '5_2', '5_3'])

    dst.fc6.W.data[:] = src.classifier.W.data[class_indices, :]
    dst.fc6.b.data[:] = src.classifier.b.data[class_indices]


def _transfer_resnet101(src, dst, class_indices):
    # Reorder weights to work on RGB and not on BGR
    dst.conv1.conv.W.data[:] = src['conv1/7x7_s2'].W.data[:, ::-1]
    dst.conv1.bn.avg_mean[:] = src['conv1/7x7_s2/bn'].avg_mean
    dst.conv1.bn.avg_var[:] = src['conv1/7x7_s2/bn'].avg_var
    dst.conv1.bn.gamma.data[:] = src['conv1/7x7_s2/bn/scale'].W.data
    dst.conv1.bn.beta.data[:] = src['conv1/7x7_s2/bn/scale'].bias.b.data

    _transfer_block(src, dst.res2, ['2_{}'.format(i) for i in range(1, 4)])
    _transfer_block(src, dst.res3, ['3_{}'.format(i) for i in range(1, 5)])
    _transfer_block(src, dst.res4, ['4_{}'.format(i) for i in range(1, 24)])
    _transfer_block(src, dst.res5, ['5_{}'.format(i) for i in range(1, 4)])

    dst.fc6.W.data[:] = src.classifier.W.data[class_indices, :]
    dst.fc6.b.data[:] = src.classifier.b.data[class_indices]


def _transfer_resnet152(src, dst, class_indices):
    # Reorder weights to work on RGB and not on BGR
    dst.conv1.conv.W.data[:] = src['conv1/7x7_s2'].W.data[:, ::-1]
    dst.conv1.bn.avg_mean[:] = src['conv1/7x7_s2/bn'].avg_mean
    dst.conv1.bn.avg_var[:] = src['conv1/7x7_s2/bn'].avg_var
    dst.conv1.bn.gamma.data[:] = src['conv1/7x7_s2/bn/scale'].W.data
    dst.conv1.bn.beta.data[:] = src['conv1/7x7_s2/bn/scale'].bias.b.data

    _transfer_block(src, dst.res2, ['2_{}'.format(i) for i in range(1, 4)])
    _transfer_block(src, dst.res3, ['3_{}'.format(i) for i in range(1, 9)])
    _transfer_block(src, dst.res4, ['4_{}'.format(i) for i in range(1, 37)])
    _transfer_block(src, dst.res5, ['5_{}'.format(i) for i in range(1, 4)])

    dst.fc6.W.data[:] = src.classifier.W.data[class_indices, :]
    dst.fc6.b.data[:] = src.classifier.b.data[class_indices]


def _load_class_indices():
    # The caffemodel weights in the original repository
    # (https://github.com/hujie-frank/SENet) have been trained with a modified
    # order of class indices.

    indices = np.zeros(1000, dtype=np.int32)
    file = os.path.join(os.path.dirname(__file__), 'label_map.csv')

    with open(file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            index_modified, index_origin = map(int, line.strip().split(','))
            indices[index_origin] = index_modified

    return indices


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_name', choices=(
            'se-resnet50', 'se-resnet101', 'se-resnet152',
            'se-resnext50', 'se-resnext101',
        ))
    parser.add_argument('caffemodel')
    parser.add_argument('output', nargs='?', default=None)
    args = parser.parse_args()

    caffemodel = CaffeFunction(args.caffemodel)
    if args.model_name == 'se-resnet50':
        model = SEResNet50(pretrained_model=None, n_class=1000)
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet50(caffemodel, model, _load_class_indices())
    elif args.model_name == 'se-resnet101':
        model = SEResNet101(pretrained_model=None, n_class=1000)
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet101(caffemodel, model, _load_class_indices())
    elif args.model_name == 'se-resnet152':
        model = SEResNet152(pretrained_model=None, n_class=1000)
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet152(caffemodel, model, _load_class_indices())
    elif args.model_name == 'se-resnext50':
        model = SEResNeXt50(pretrained_model=None, n_class=1000)
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet50(caffemodel, model, _load_class_indices())
    elif args.model_name == 'se-resnext101':
        model = SEResNeXt101(pretrained_model=None, n_class=1000)
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet101(caffemodel, model, _load_class_indices())

    if args.output is None:
        output = '{}_imagenet_convert.npz'.format(args.model_name)
    else:
        output = args.output
    chainer.serializers.save_npz(output, model)


if __name__ == '__main__':
    main()
