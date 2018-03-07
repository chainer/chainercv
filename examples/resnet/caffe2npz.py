import argparse

import numpy as np

import chainer
from chainer.links.caffe.caffe_function import CaffeFunction

from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
    src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
    src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
    dst_conv.W.data[:] = src_conv.W.data
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src, dst.conv1.conv, dst.conv1.bn, name, '2a')
    _transfer_components(src, dst.conv2.conv, dst.conv2.bn, name, '2b')
    _transfer_components(src, dst.conv3.conv, dst.conv3.bn, name, '2c')
    _transfer_components(
        src, dst.residual_conv.conv, dst.residual_conv.bn, name, '1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src, dst.conv1.conv, dst.conv1.bn, name, '2a')
    _transfer_components(src, dst.conv2.conv, dst.conv2.bn, name, '2b')
    _transfer_components(src, dst.conv3.conv, dst.conv3.bn, name, '2c')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet50(src, dst):
    # Reorder weights to work on RGB and not on BGR
    dst.conv1.conv.W.data[:] = src.conv1.W.data[:, ::-1]
    dst.conv1.conv.b.data[:] = src.conv1.b.data
    dst.conv1.bn.avg_mean[:] = src.bn_conv1.avg_mean
    dst.conv1.bn.avg_var[:] = src.bn_conv1.avg_var
    dst.conv1.bn.gamma.data[:] = src.scale_conv1.W.data
    dst.conv1.bn.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    _transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet101(src, dst):
    # Reorder weights to work on RGB and not on BGR
    dst.conv1.conv.W.data[:] = src.conv1.W.data[:, ::-1]
    dst.conv1.bn.avg_mean[:] = src.bn_conv1.avg_mean
    dst.conv1.bn.avg_var[:] = src.bn_conv1.avg_var
    dst.conv1.bn.gamma.data[:] = src.scale_conv1.W.data
    dst.conv1.bn.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b1', '3b2', '3b3'])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 23)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet152(src, dst):
    # Reorder weights to work on RGB and not on BGR
    dst.conv1.conv.W.data[:] = src.conv1.W.data[:, ::-1]
    dst.conv1.bn.avg_mean[:] = src.bn_conv1.avg_mean
    dst.conv1.bn.avg_var[:] = src.bn_conv1.avg_var
    dst.conv1.bn.gamma.data[:] = src.scale_conv1.W.data
    dst.conv1.bn.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3,
                    ['3a'] + ['3b{}'.format(i) for i in range(1, 8)])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 36)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_name', choices=('resnet50', 'resnet101', 'resnet152'))
    parser.add_argument('caffemodel')
    parser.add_argument('output', nargs='?', default=None)
    args = parser.parse_args()

    caffemodel = CaffeFunction(args.caffemodel)
    if args.model_name == 'resnet50':
        model = ResNet50(pretrained_model=None, n_class=1000, arch='he')
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet50(caffemodel, model)
    elif args.model_name == 'resnet101':
        model = ResNet101(pretrained_model=None, n_class=1000, arch='he')
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet101(caffemodel, model)
    elif args.model_name == 'resnet152':
        model = ResNet152(pretrained_model=None, n_class=1000, arch='he')
        model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        _transfer_resnet152(caffemodel, model)

    if args.output is None:
        output = '{}_imagenet_convert.npz'.format(args.model_name)
    else:
        output = args.output
    chainer.serializers.save_npz(output, model)


if __name__ == '__main__':
    main()
