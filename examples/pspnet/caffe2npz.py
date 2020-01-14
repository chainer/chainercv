import argparse
from google.protobuf import text_format
import numpy as np
import re

from chainer import serializers
from chainercv.experimental.links import PSPNetResNet101

import caffe_pb2


def copy_conv(layer, config, conv, has_bias=False, inverse_ch=False):
    data = np.array(layer.blobs[0].data)
    conv.W.data.ravel()[:] = data
    if inverse_ch:
        conv.W.data[:] = conv.W.data[:, ::-1, ...]
    if has_bias:
        conv.b.data[:] = np.array(layer.blobs[1].data)
    return conv


def copy_conv2d_bn_activ(layer, config, cba, inverse_ch=False):
    if 'Convolution' in layer.type:
        cba.conv = copy_conv(layer, config, cba.conv, inverse_ch=inverse_ch)
    elif 'BN' in layer.type:
        cba.bn.eps = config.bn_param.eps
        cba.bn.decay = config.bn_param.momentum
        cba.bn.gamma.data.ravel()[:] = np.array(layer.blobs[0].data)
        cba.bn.beta.data.ravel()[:] = np.array(layer.blobs[1].data)
        cba.bn.avg_mean.ravel()[:] = np.array(layer.blobs[2].data)
        cba.bn.avg_var.ravel()[:] = np.array(layer.blobs[3].data)
    else:
        print('Ignored: {} ({})'.format(layer.name, layer.type))
    return cba


def copy_res1(layer, config, block):
    if layer.name.startswith('conv1_1'):
        block.conv1_1 = copy_conv2d_bn_activ(
            layer, config, block.conv1_1, inverse_ch=True)
    elif layer.name.startswith('conv1_2'):
        block.conv1_2 = copy_conv2d_bn_activ(layer, config, block.conv1_2)
    elif layer.name.startswith('conv1_3'):
        block.conv1_3 = copy_conv2d_bn_activ(layer, config, block.conv1_3)
    else:
        print('Ignored: {} ({})'.format(layer.name, layer.type))
    return block


def copy_bottleneck(layer, config, block):
    if 'reduce' in layer.name:
        block.conv1 = copy_conv2d_bn_activ(layer, config, block.conv1)
    elif '3x3' in layer.name:
        block.conv2 = copy_conv2d_bn_activ(layer, config, block.conv2)
    elif 'increase' in layer.name:
        block.conv3 = copy_conv2d_bn_activ(layer, config, block.conv3)
    elif 'proj' in layer.name:
        block.residual_conv = copy_conv2d_bn_activ(
            layer, config, block.residual_conv)
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
    pool_id = int(ret.groups()[0])
    linear_id = {1: 3,
                 2: 2,
                 3: 1,
                 6: 0}[pool_id]
    block._children[linear_id] =\
        copy_conv2d_bn_activ(layer, config, block[linear_id])
    return block


def transfer(model, param, net):
    name_config = dict([(l.name, l) for l in net.layer])
    for layer in param.layer:
        if layer.name not in name_config:
            continue
        config = name_config[layer.name]
        if layer.name.startswith('conv1'):
            model.extractor = copy_res1(layer, config, model.extractor)
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
            model.head_conv1 = copy_conv2d_bn_activ(
                layer, config, model.head_conv1)
        elif layer.name.startswith('conv6'):
            model.head_conv2 = copy_conv(
                layer, config, model.head_conv2, has_bias=True)
        # NOTE: Auxirillary is not copied
        else:
            print('Ignored: {} ({})'.format(layer.name, layer.type))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel')
    parser.add_argument('output')
    args = parser.parse_args()

    proto_path = 'weights/pspnet101_cityscapes_713.prototxt'

    model = PSPNetResNet101(**PSPNetResNet101.preset_params['cityscapes'])
    input_size = PSPNetResNet101.preset_params['cityscapes']['input_size']
    model(np.random.uniform(size=(1, 3) + input_size).astype(np.float32))

    caffe_param = caffe_pb2.NetParameter()
    caffe_param.MergeFromString(open(args.caffemodel, 'rb').read())
    caffe_net = text_format.Merge(
        open(proto_path).read(), caffe_pb2.NetParameter())

    transfer(model, caffe_param, caffe_net)
    serializers.save_npz(args.output, model)


if __name__ == '__main__':
    main()
