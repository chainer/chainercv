from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet import ResBlock
from chainercv.links import PickableSequentialChain
from chainercv import utils


# RGB order
# This is channel wise mean of mean image distributed at
# https://github.com/KaimingHe/deep-residual-networks
_imagenet_mean = np.array(
    [123.15163084, 115.90288257, 103.0626238],
    dtype=np.float32)[:, np.newaxis, np.newaxis]


class SEResNet(PickableSequentialChain):

    """Base class for SE-ResNet architecture.

    This architecture is based on ResNet. A squeeze-and-excitation block is
    applied at the end of each non-identity branch of residual block. Please
    refer to `the original paper  <https://arxiv.org/pdf/1709.01507.pdf>`_
    for a detailed description of network architecture.

    Similar to :class:`chainercv.links.model.resnet.ResNet`, ImageNet
    pretrained weights are downloaded when :obj:`pretrained_model` argument
    is :obj:`imagenet`, originally distributed at `the Github repository by
    one of the paper authors <https://github.com/hujie-frank/SENet>`_.

    .. seealso::
        :class:`chainercv.links.model.resnet.ResNet`
        :class:`chainercv.links.connection.SEBlock`

    Args:
        n_layer (int): The number of layers.
        n_class (int): The number of classes. If :obj:`None`,
            the default values are used.
            If a supported pretrained model is used,
            the number of classes used to train the pretrained model
            is used. Otherwise, the number of classes in ILSVRC 2012 dataset
            is used.
        pretrained_model (string): The destination of the pre-trained
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

    """

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    _models = {
        50: {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean},
                'overwritable': {'mean'},
                'url': 'https://chainercv-models.preferred.jp/'
                'se_resnet50_imagenet_converted_2018_06_25.npz'
            },
        },
        101: {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean},
                'overwritable': {'mean'},
                'url': 'https://chainercv-models.preferred.jp/'
                'se_resnet101_imagenet_converted_2018_06_25.npz'
            },
        },
        152: {
            'imagenet': {
                'param': {'n_class': 1000, 'mean': _imagenet_mean},
                'overwritable': {'mean'},
                'url': 'https://chainercv-models.preferred.jp/'
                'se_resnet152_imagenet_converted_2018_06_25.npz'
            },
        }
    }

    def __init__(self, n_layer,
                 n_class=None,
                 pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        blocks = self._blocks[n_layer]

        param, path = utils.prepare_pretrained_model(
            {'n_class': n_class, 'mean': mean},
            pretrained_model, self._models[n_layer],
            {'n_class': 1000, 'mean': _imagenet_mean})
        self.mean = param['mean']

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initializers.Normal(scale=0.01)
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            initialW = initializers.constant.Zero()
            fc_kwargs['initialW'] = initializers.constant.Zero()
        kwargs = {
            'initialW': initialW, 'stride_first': True, 'add_seblock': True}

        super(SEResNet, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 7, 2, 3, nobias=True,
                                       initialW=initialW)
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.res2 = ResBlock(blocks[0], None, 64, 256, 1, **kwargs)
            self.res3 = ResBlock(blocks[1], None, 128, 512, 2, **kwargs)
            self.res4 = ResBlock(blocks[2], None, 256, 1024, 2, **kwargs)
            self.res5 = ResBlock(blocks[3], None, 512, 2048, 2, **kwargs)
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(None, param['n_class'], **fc_kwargs)
            self.prob = F.softmax

        if path:
            chainer.serializers.load_npz(path, self)


class SEResNet50(SEResNet):

    """SE-ResNet-50 Network.

    Please consult the documentation for :class:`SEResNet`.

    .. seealso::
        :class:`chainercv.links.model.senet.SEResNet`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNet50, self).__init__(
            50, n_class, pretrained_model,
            mean, initialW, fc_kwargs)


class SEResNet101(SEResNet):

    """SE-ResNet-101 Network.

    Please consult the documentation for :class:`SEResNet`.

    .. seealso::
        :class:`chainercv.links.model.senet.SEResNet`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNet101, self).__init__(
            101, n_class, pretrained_model,
            mean, initialW, fc_kwargs)


class SEResNet152(SEResNet):

    """SE-ResNet-152 Network.

    Please consult the documentation for :class:`SEResNet`.

    .. seealso::
        :class:`chainercv.links.model.senet.SEResNet`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNet152, self).__init__(
            152, n_class, pretrained_model,
            mean, initialW, fc_kwargs)
