from __future__ import print_function

import argparse
import os
import re

import chainer
import chainer.links as L
from chainer import serializers
from chainercv.links import PSPNetResNet101
from google.protobuf import text_format
import numpy as np

import caffe_pb2


def get_chainer_model(n_class, input_size, n_blocks, pyramids, mid_stride):
    with chainer.using_config('train', True):
        model = PSPNetResNet101(
            n_class, None, input_size)
        model(np.random.rand(1, 3, input_size, input_size).astype(np.float32))
    size = 0
    for param in model.params():
        try:
            size += param.size
        except Exception as e:
            print(str(type(e)), e, param, param.name)
            exit(-1)
    print('PSPNet (chainer) size:', size)
    return model


def get_param_net(prodo_dir, param_fn, proto_fn):
    print('Loading caffe parameters...', end=' ')
    param = caffe_pb2.NetParameter()
    param.MergeFromString(open(param_fn, 'rb').read())
    print('done')

    print('Loading caffe prototxt...', end=' ')
    proto_fp = open(proto_fn).read()
    net = caffe_pb2.NetParameter()
    net = text_format.Merge(proto_fp, net)
    print('done')

    return param, net


def copy_conv(layer, config, conv, has_bias=False, inverse_ch=False):
    data = np.array(layer.blobs[0].data)
    if inverse_ch:
        conv.W.data[:] = data.reshape(conv.W.shape)
        conv.W.data[:] = conv.W.data[:, ::-1, ...]
    else:
        conv.W.data[:] = data.reshape(conv.W.shape)
    if has_bias:
        data = np.array(layer.blobs[1].data)
        conv.b.data[:] = data.reshape(conv.b.shape)

    # Check ksize
    assert config.convolution_param.kernel_size[0] == conv.ksize, \
        'ksize: {} != {} ({}, {}, {}, {})'.format(
            config.convolution_param.kernel_size[0], conv.ksize,
            layer.name, config, conv, conv.name)

    # Check stride
    if len(config.convolution_param.stride) == 1:
        stride = config.convolution_param.stride[0]
        stride = (stride, stride)
    assert stride == conv.stride, \
        'stride: {} != {} ({}, {}, {}, {})'.format(
            stride, conv.stride, layer.name, config, conv, conv.name)

    # Check pad
    if len(config.convolution_param.pad) == 1:
        pad = config.convolution_param.pad[0]
        pad = (pad, pad)
    elif config.convolution_param.pad == []:
        pad = (0, 0)
    assert pad == conv.pad, \
        'pad: {} != {} ({}, {}, {}, {})'.format(
            pad, conv.pad, layer.name, config, conv, conv.name)

    assert layer.convolution_param.bias_term == has_bias
    if not has_bias:
        assert conv.b is None
    if isinstance(config.convolution_param.dilation, int):
        assert isinstance(conv, L.DilatedConvolution2D)
        assert config.convolution_param.dilation == conv.dilate

    return conv


def copy_cbr(layer, config, cbr, inverse_ch=False):
    if 'Convolution' in layer.type:
        cbr.conv = copy_conv(layer, config, cbr.conv, inverse_ch=inverse_ch)
    elif 'BN' in layer.type:
        cbr.bn.eps = config.bn_param.eps
        cbr.bn.decay = config.bn_param.momentum
        cbr.bn.gamma.data.ravel()[:] = np.array(layer.blobs[0].data).ravel()
        cbr.bn.beta.data.ravel()[:] = np.array(layer.blobs[1].data).ravel()
        cbr.bn.avg_mean.ravel()[:] = np.array(layer.blobs[2].data).ravel()
        cbr.bn.avg_var.ravel()[:] = np.array(layer.blobs[3].data).ravel()
    else:
        print('Ignored: {} ({})'.format(layer.name, layer.type))
    return cbr


def copy_conv2d_bn_activ(layer, config, cba, inverse_ch=False):
    if 'Convolution' in layer.type:
        cba.conv = copy_conv(layer, config, cba.conv, inverse_ch=inverse_ch)
    elif 'BN' in layer.type:
        cba.bn.eps = config.bn_param.eps
        cba.bn.decay = config.bn_param.momentum
        cba.bn.gamma.data.ravel()[:] = np.array(layer.blobs[0].data).ravel()
        cba.bn.beta.data.ravel()[:] = np.array(layer.blobs[1].data).ravel()
        cba.bn.avg_mean.ravel()[:] = np.array(layer.blobs[2].data).ravel()
        cba.bn.avg_var.ravel()[:] = np.array(layer.blobs[3].data).ravel()
    else:
        print('Ignored: {} ({})'.format(layer.name, layer.type))
    return cba


def copy_head(layer, config, block):
    if layer.name.startswith('conv1_1'):
        # You do not need this for VOC2012
        block.conv1_1 = copy_cbr(layer, config, block.conv1_1, inverse_ch=True)
    elif layer.name.startswith('conv1_2'):
        block.conv1_2 = copy_cbr(layer, config, block.conv1_2)
    elif layer.name.startswith('conv1_3'):
        block.conv1_3 = copy_cbr(layer, config, block.conv1_3)
    else:
        print('Ignored: {} ({})'.format(layer.name, layer.type))
    return block


def copy_bottleneck(layer, config, block):
    if 'reduce' in layer.name:
        block.conv1 = copy_cbr(layer, config, block.conv1)
    elif '3x3' in layer.name:
        block.conv2 = copy_cbr(layer, config, block.conv2)
    elif 'increase' in layer.name:
        block.conv3 = copy_cbr(layer, config, block.conv3)
    elif 'proj' in layer.name:
        block.residual_conv = copy_cbr(layer, config, block.residual_conv)
    else:
        print('Ignored: {} ({})'.format(layer.name, layer.type))
    return block


def copy_resblock(layer, config, block):
    if '/' in layer.name:
        layer.name = layer.name.split('/')[0]
    i = int(layer.name.split('_')[1]) - 1
    if i == 0:
        name = 'a'
    else:
        name = 'b{}'.format(i)
    setattr(block, name,
            copy_bottleneck(layer, config, getattr(block, name)))
    return block


def copy_ppm_module(layer, config, block):
    ret = re.search('pool([0-9]+)', layer.name)
    if ret is None:
        raise ValueError('Error in copy_ppm_module:'
                         '{}, {}, {}'.format(layer.name, config, block))
    i = int(ret.groups()[0])
    i = {1: 3,
         2: 2,
         3: 1,
         6: 0}[i]
    block._children[i] = copy_conv2d_bn_activ(layer, config, block[i])
    return block


def transfer(model, param, net):
    name_config = dict([(l.name, l) for l in net.layer])
    for layer in param.layer:
        if layer.name not in name_config:
            continue
        config = name_config[layer.name]
        if layer.name.startswith('conv1'):
            model.extractor = copy_head(layer, config, model.extractor)
        elif layer.name.startswith('conv2'):
            model.extractor.res2 = copy_resblock(
                layer, config, model.extractor.res2)
        elif layer.name.startswith('conv3'):
            model.extractor.res3 = copy_resblock(
                layer, config, model.extractor.res3)
        elif layer.name.startswith('conv4'):
            model.extractor.res4 = copy_resblock(
                layer, config, model.extractor.res4)
        elif layer.name.startswith('conv5') \
                and 'pool' not in layer.name \
                and 'conv5_4' not in layer.name:
            model.extractor.res5 = copy_resblock(
                layer, config, model.extractor.res5)
        elif layer.name.startswith('conv5_3') and 'pool' in layer.name:
            model.ppm = copy_ppm_module(layer, config, model.ppm)
        elif layer.name.startswith('conv5_4'):
            model.head_conv1 = copy_cbr(layer, config, model.head_conv1)
        elif layer.name.startswith('conv6'):
            model.head_conv2 = copy_conv(
                layer, config, model.head_conv2, has_bias=True)
        # NOTE: Auxirillary is not copied
        else:
            print('Ignored: {} ({})'.format(layer.name, layer.type))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel')
    parser.add_argument('output')
    args = parser.parse_args()

    proto_dir = 'weights'

    settings = {
        'cityscapes': {
            'proto_fn': 'pspnet101_cityscapes_713.prototxt',
            'param_fn': 'pspnet101_cityscapes.caffemodel',
            'n_class': 19,
            'input_size': 713,
            'n_blocks': [3, 4, 23, 3],
            'feat_size': 90,
            'mid_stride': True,
            'pyramids': [6, 3, 2, 1],
        },
    }

    dataset_name = 'cityscapes'
    proto_fn = settings[dataset_name]['proto_fn']
    param_fn = settings[dataset_name]['param_fn']
    n_class = settings[dataset_name]['n_class']
    input_size = settings[dataset_name]['input_size']
    n_blocks = settings[dataset_name]['n_blocks']
    pyramids = settings[dataset_name]['pyramids']
    mid_stride = settings[dataset_name]['mid_stride']

    name = os.path.splitext(proto_fn)[0]
    param_fn = os.path.join(proto_dir, param_fn)
    proto_fn = os.path.join(proto_dir, proto_fn)

    model = get_chainer_model(
        n_class, input_size, n_blocks, pyramids, mid_stride)
    param, net = get_param_net(proto_dir, args.caffemodel, proto_fn)
    model = transfer(model, param, net)

    serializers.save_npz(args.output, model)
