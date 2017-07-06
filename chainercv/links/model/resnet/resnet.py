from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links.model.resnet.building_block import BuildingBlock
from chainercv.links import SequentialFeatureExtractor
from chainercv.utils import download_model


# RGB order
# This is channel wise mean of mean image distributed at
# https://github.com/KaimingHe/deep-residual-networks
_imagenet_mean = np.array(
    [123.15163084, 115.90288257, 103.0626238],
    dtype=np.float32)[:, np.newaxis, np.newaxis]


class ResNet(SequentialFeatureExtractor):

    _blocks = {
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3]
    }

    _models = {
        'resnet50': {
            'imagenet': {
                'n_class': 1000,
                'url': 'https://github.com/yuyu2172/share-weights/releases/'
                'download/0.0.3/resnet50_imagenet_convert_2017_07_06.npz',
                'mean': _imagenet_mean
            },
        },
        'resnet101': {
            'imagenet': {
                'n_class': 1000,
                'url': 'https://github.com/yuyu2172/share-weights/releases/'
                'download/0.0.3/resnet101_imagenet_convert_2017_07_06.npz',
                'mean': _imagenet_mean
            },
        },
        'resnet152': {
            'imagenet': {
                'n_class': 1000,
                'url': 'https://github.com/yuyu2172/share-weights/releases/'
                'download/0.0.3/resnet152_imagenet_convert_2017_07_06.npz',
                'mean': _imagenet_mean
            },
        }
    }

    def __init__(self, model_name,
                 pretrained_model=None,
                 n_class=None,
                 mean=None, initialW=None):
        block = self._blocks[model_name]
        _models = self._models[model_name]
        if n_class is None:
            if pretrained_model in self._models:
                n_class = _models[pretrained_model]['n_class']
            else:
                n_class = 1000

        if mean is None:
            if pretrained_model in _models:
                mean = _models[pretrained_model]['mean']
        self.mean = mean

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            if initialW is None:
                initialW = initializers.constant.Zero()
        else:
            # Employ default initializers used in the original paper.
            if initialW is None:
                initialW = initializers.normal.HeNormal(scale=1.)
        kwargs = {'initialW': initialW}

        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, 2, 3, **kwargs)
            self.bn1 = L.BatchNormalization(64)
            self.conv1_relu = F.relu
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.res2 = BuildingBlock(block[0], None, 64, 256, 1, **kwargs)
            self.res3 = BuildingBlock(block[1], None, 128, 512, 2, **kwargs)
            self.res4 = BuildingBlock(block[2], None, 256, 1024, 2, **kwargs)
            self.res5 = BuildingBlock(block[3], None, 512, 2048, 2, **kwargs)
            self.pool5 = _global_average_pooling_2d
            self.fc6 = L.Linear(None, n_class)
            self.prob = F.softmax

        if pretrained_model in _models:
            path = download_model(_models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = h.reshape(n, channel)
    return h


class ResNet50(ResNet):

    def __init__(self, pretrained_model=None,
                 n_class=None, mean=None, initialW=None):
        super(ResNet50, self).__init__(
            'resnet50', pretrained_model,
            n_class, mean, initialW)


class ResNet101(ResNet):

    def __init__(self, pretrained_model=None,
                 n_class=None, mean=None, initialW=None):
        super(ResNet101, self).__init__(
            'resnet101', pretrained_model,
            n_class, mean, initialW)


class ResNet152(ResNet):

    def __init__(self, pretrained_model=None,
                 n_class=None, mean=None, initialW=None):
        super(ResNet152, self).__init__(
            'resnet152', pretrained_model,
            n_class, mean, initialW)
