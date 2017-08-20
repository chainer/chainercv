import argparse

import chainer
import chainer.links.caffe.caffe_function as caffe

from chainercv.links import VGG16

"""
Please download a weight from here.
http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/
caffe/VGG_ILSVRC_16_layers.caffemodel
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel')
    parser.add_argument('output')
    args = parser.parse_args()

    caffemodel = caffe.CaffeFunction(args.caffemodel)
    model = VGG16(pretrained_model=None, n_class=1000)

    model.conv1_1.conv.copyparams(caffemodel.conv1_1)
    # The pretrained weights are trained to accept BGR images.
    # Convert weights so that they accept RGB images.
    model.conv1_1.conv.W.data[:] = model.conv1_1.conv.W.data[:, ::-1]
    model.conv1_2.conv.copyparams(caffemodel.conv1_2)
    model.conv2_1.conv.copyparams(caffemodel.conv2_1)
    model.conv2_2.conv.copyparams(caffemodel.conv2_2)
    model.conv3_1.conv.copyparams(caffemodel.conv3_1)
    model.conv3_2.conv.copyparams(caffemodel.conv3_2)
    model.conv3_3.conv.copyparams(caffemodel.conv3_3)
    model.conv4_1.conv.copyparams(caffemodel.conv4_1)
    model.conv4_2.conv.copyparams(caffemodel.conv4_2)
    model.conv4_3.conv.copyparams(caffemodel.conv4_3)
    model.conv5_1.conv.copyparams(caffemodel.conv5_1)
    model.conv5_2.conv.copyparams(caffemodel.conv5_2)
    model.conv5_3.conv.copyparams(caffemodel.conv5_3)
    model.fc6.copyparams(caffemodel.fc6)
    model.fc7.copyparams(caffemodel.fc7)
    model.fc8.copyparams(caffemodel.fc8)
    chainer.serializers.save_npz(args.output, model)


if __name__ == '__main__':
    main()
