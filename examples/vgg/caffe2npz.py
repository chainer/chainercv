import argparse
import re

import chainer
from chainer import Link
import chainer.links.caffe.caffe_function as caffe


"""
Please download a weight from here.
http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
"""


def rename(name):
    m = re.match(r'conv(\d+)_(\d+)$', name)
    if m:
        i, j = map(int, m.groups())
        return 'conv{:d}_{:d}/conv'.format(i, j)

    return name


class VGGCaffeFunction(caffe.CaffeFunction):

    def __init__(self, model_path):
        print('loading weights from {:s} ... '.format(model_path))
        super(VGGCaffeFunction, self).__init__(model_path)

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, Link):
            new_name = rename(name)

            if new_name == 'conv1_1/conv':
                # BGR -> RGB
                value.W.array[:, ::-1] = value.W.array
                print('{:s} -> {:s} (BGR -> RGB)'.format(name, new_name))
            else:
                print('{:s} -> {:s}'.format(name, new_name))
        else:
            new_name = name

        super(VGGCaffeFunction, self).__setattr__(new_name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel')
    parser.add_argument('output')
    args = parser.parse_args()

    model = VGGCaffeFunction(args.caffemodel)
    chainer.serializers.save_npz(args.output, model)


if __name__ == '__main__':
    main()
