import argparse
import numpy as np
import re

import chainer.links.caffe.caffe_function as caffe
from chainer import serializers

from chainercv.links.model.ssd import Normalize


def rename(name):
    m = re.match(r'^conv(\d+)_([123])$', name)
    if m:
        i, j = map(int, m.groups())
        if i >= 6:
            i += 2
        return 'extractor/conv{:d}_{:d}'.format(i, j)

    m = re.match(r'^fc([67])$', name)
    if m:
        return 'extractor/conv{:d}'.format(int(m.group(1)))

    if name == r'conv4_3_norm':
        return 'extractor/norm4'

    m = re.match(r'^conv4_3_norm_mbox_(loc|conf)$', name)
    if m:
        return 'multibox/{:s}/0'.format(m.group(1))

    m = re.match(r'^fc7_mbox_(loc|conf)$', name)
    if m:
        return ('multibox/{:s}/1'.format(m.group(1)))

    m = re.match(r'^conv(\d+)_2_mbox_(loc|conf)$', name)
    if m:
        i, type_ = int(m.group(1)), m.group(2)
        if i >= 6:
            return 'multibox/{:s}/{:d}'.format(type_, i - 4)

    return name


class SSDCaffeFunction(caffe.CaffeFunction):

    def __init__(self, model_path):
        print('loading weights from {:s} ... '.format(model_path))
        super(SSDCaffeFunction, self).__init__(model_path)

    def add_link(self, name, link):
        new_name = rename(name)
        print('{:s} -> {:s}'.format(name, new_name))
        super(SSDCaffeFunction, self).add_link(new_name, link)

    @caffe._layer('Normalize', None)
    def _setup_normarize(self, layer):
        blobs = layer.blobs
        func = Normalize(caffe._get_num(blobs[0]))
        func.scale.data[:] = np.array(blobs[0].data)
        self.add_link(layer.name, func)

    @caffe._layer('AnnotatedData', None)
    @caffe._layer('Flatten', None)
    @caffe._layer('MultiBoxLoss', None)
    @caffe._layer('Permute', None)
    @caffe._layer('PriorBox', None)
    def _skip_layer(self, _):
        pass


def convert_xy_conv(l):
    b = l.b.data.reshape(-1, 4)
    b_old_x = b[:, [0, 2]]
    b_old_y = b[:, [1, 3]]
    b[:, [0, 2]] = b_old_y
    b[:, [1, 3]] = b_old_x

    out_C, in_C, kh, kw = l.W.shape
    W = l.W.data.reshape(-1, 4, in_C, kh, kw)
    W_old_x = W[:, [0, 2]]
    W_old_y = W[:, [1, 3]]
    W[:, [0, 2]] = W_old_y
    W[:, [1, 3]] = W_old_x

    l.b.data[:] = b.reshape(-1)
    l.W.data[:] = W.reshape(-1, in_C, kh, kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel')
    parser.add_argument('output')
    args = parser.parse_args()

    model = SSDCaffeFunction(args.caffemodel)
    # The pretrained weights are trained to accept BGR images.
    # Convert weights so that they accept RGB images.
    model['extractor/conv1_1'].W.data[:] =\
        model['extractor/conv1_1'].W.data[:, ::-1]

    # The pretrained model outputs coordinates in xy convention.
    # This needs to be changed to yx convention, which is used
    # in ChainerCV.
    for name in sorted([child.name for child in model.children()]):
        if name[:12] == 'multibox/loc':
            convert_xy_conv(model[name])

    serializers.save_npz(args.output, model)


if __name__ == '__main__':
    main()
