from __future__ import division

import numpy as np
import warnings

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet.resblock import ResBlock
from chainercv.links import PickableSequentialChain
from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


# RGB order
# This is channel wise mean of mean image distributed at
# https://github.com/KaimingHe/deep-residual-networks
imagenet_he_mean = np.array(
    [123.15163084, 115.90288257, 103.0626238],
    dtype=np.float32)[:, np.newaxis, np.newaxis]


class ResNet(PickableSequentialChain):

    """Base class for ResNet architecture.

    This is a pickable sequential link.
    The network can choose output layers from set of all
    intermediate layers.
    The attribute :obj:`pick` is the names of the layers that are going
    to be picked by :meth:`__call__`.
    The attribute :obj:`layer_names` is the names of all layers
    that can be picked.

    Examples:

        >>> model = ResNet50()
        # By default, __call__ returns a probability score (after Softmax).
        >>> prob = model(imgs)
        >>> model.pick = 'res5'
        # This is layer res5
        >>> res5 = model(imgs)
        >>> model.pick = ['res5', 'fc6']
        >>> # These are layers res5 and fc6.
        >>> res5, fc6 = model(imgs)

    .. seealso::
        :class:`chainercv.links.model.PickableSequentialChain`

    When :obj:`pretrained_model` is the path of a pre-trained chainer model
    serialized as a :obj:`.npz` file in the constructor, this chain model
    automatically initializes all the parameters with it.
    When a string in the prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`imagenet`: Loads weights trained with ImageNet and distributed \
        at `Model Zoo \
        <https://github.com/BVLC/caffe/wiki/Model-Zoo>`_.
        This is only supported when :obj:`arch=='he'`.

    Args:
        n_layer (int): The number of layers.
        n_class (int): The number of classes. If :obj:`None`,
            the default values are used.
            If a supported pretrained model is used,
            the number of classes used to train the pretrained model
            is used. Otherwise, the number of classes in ILSVRC 2012 dataset
            is used.
        pretrained_model (str): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        mean (numpy.ndarray): A mean value. If :obj:`None`,
            the default values are used.
            If a supported pretrained model is used,
            the mean value used to train the pretrained model is used.
            Otherwise, the mean value calculated from ILSVRC 2012 dataset
            is used.
        initialW (callable): Initializer for the weights of
            convolution kernels.
        fc_kwargs (dict): Keyword arguments passed to initialize
            the :class:`chainer.links.Linear`.
        arch (str): If :obj:`fb`, use Facebook ResNet
            architecture. When :obj:`he`, use the architecture presented
            by `the original ResNet paper \
            <https://arxiv.org/pdf/1512.03385.pdf>`_.
            This option changes where to apply strided convolution.
            The default value is :obj:`fb`.

    """

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    _models = {
        'fb': {
            50: {},
            101: {},
            152: {}
        },
        'he': {
            50: {
                'imagenet': {
                    'n_class': 1000,
                    'url': 'https://github.com/yuyu2172/share-weights/'
                    'releases/download/0.0.6/'
                    'resnet50_imagenet_convert_2018_03_07.npz',
                    'mean': imagenet_he_mean
                },
            },
            101: {
                'imagenet': {
                    'n_class': 1000,
                    'url': 'https://github.com/yuyu2172/share-weights/'
                    'releases/download/0.0.6/'
                    'resnet101_imagenet_convert_2018_03_07.npz',
                    'mean': imagenet_he_mean
                },
            },
            152: {
                'imagenet': {
                    'n_class': 1000,
                    'url': 'https://github.com/yuyu2172/share-weights/'
                    'releases/download/0.0.6/'
                    'resnet152_imagenet_convert_2018_03_07.npz',
                    'mean': imagenet_he_mean
                },
            }
        }
    }

    def __init__(self, n_layer,
                 n_class=None,
                 pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}, arch='fb'):
        if arch == 'fb':
            if pretrained_model == 'imagenet':
                raise ValueError(
                    'Pretrained weights for Facebook ResNet models '
                    'are not supported. Please set mode to \'he\'.')
            stride_first = False
            conv1_no_bias = True
        elif arch == 'he':
            stride_first = True
            conv1_no_bias = False
        else:
            raise ValueError('arch is expected to be one of [\'he\', \'fb\']')
        _models = self._models[arch][n_layer]
        blocks = self._blocks[n_layer]

        if n_class is None:
            if pretrained_model in _models:
                n_class = _models[pretrained_model]['n_class']
            else:
                n_class = 1000

        if mean is None:
            if pretrained_model in _models:
                mean = _models[pretrained_model]['mean']
            else:
                mean = imagenet_he_mean
        self.mean = mean

        if initialW is None:
            initialW = HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initializers.Normal(scale=0.01)
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            initialW = initializers.constant.Zero()
            fc_kwargs['initialW'] = initializers.constant.Zero()
        kwargs = {'initialW': initialW, 'stride_first': stride_first}

        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 7, 2, 3, nobias=conv1_no_bias,
                                       initialW=initialW)
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.res2 = ResBlock(blocks[0], None, 64, 256, 1, **kwargs)
            self.res3 = ResBlock(blocks[1], None, 128, 512, 2, **kwargs)
            self.res4 = ResBlock(blocks[2], None, 256, 1024, 2, **kwargs)
            self.res5 = ResBlock(blocks[3], None, 512, 2048, 2, **kwargs)
            self.pool5 = _global_average_pooling_2d
            self.fc6 = L.Linear(None, n_class, **fc_kwargs)
            self.prob = F.softmax

        if pretrained_model in _models:
            if not _available:
                warnings.warn('cv2 is not installed on your environment. '
                              'The scores of ResNets reported in the '
                              'README of the ChainerCV\'s classification '
                              'example are calculated using OpenCV as the '
                              'backend. With Pillow as the '
                              'backend, the scores would change.',
                              RuntimeWarning)
            path = download_model(_models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = h.reshape((n, channel))
    return h


class ResNet50(ResNet):

    """ResNet-50 Network.

    Please consult the documentation for :class:`ResNet`.

    .. seealso::
        :class:`chainercv.links.model.resnet.ResNet`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}, arch='fb'):
        super(ResNet50, self).__init__(
            50, n_class, pretrained_model,
            mean, initialW, fc_kwargs, arch)


class ResNet101(ResNet):

    """ResNet-101 Network.

    Please consult the documentation for :class:`ResNet`.

    .. seealso::
        :class:`chainercv.links.model.resnet.ResNet`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}, arch='fb'):
        super(ResNet101, self).__init__(
            101, n_class, pretrained_model,
            mean, initialW, fc_kwargs, arch)


class ResNet152(ResNet):

    """ResNet-152 Network.

    Please consult the documentation for :class:`ResNet`.

    .. seealso::
        :class:`chainercv.links.model.resnet.ResNet`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}, arch='fb'):
        super(ResNet152, self).__init__(
            152, n_class, pretrained_model,
            mean, initialW, fc_kwargs, arch)


class HeNormal(chainer.initializer.Initializer):

    # fan_option is not supported in Chainer v3.
    # Related: https://github.com/chainer/chainer/pull/3482

    def __init__(self, scale=1.0, dtype=None, fan_option='fan_in'):
        self.scale = scale
        self.fan_option = fan_option
        super(HeNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, fan_out = chainer.initializer.get_fans(array.shape)
        if self.fan_option == 'fan_in':
            s = self.scale * np.sqrt(2. / fan_in)
        elif self.fan_option == 'fan_out':
            s = self.scale * np.sqrt(2. / fan_out)
        else:
            raise ValueError(
                'fan_option should be either \'fan_in\' or \'fan_out\'.')
        initializers.Normal(s)(array)
