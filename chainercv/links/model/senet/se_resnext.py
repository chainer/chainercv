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


class SEResNeXt(PickableSequentialChain):

    """Base class for SE-ResNeXt architecture.

    ResNeXt is a ResNet-based architecture, where grouped convolution is
    adopted to the second convolution layer of each bottleneck block.
    In addition, a squeeze-and-excitation block is applied at the end of
    each non-identity branch of residual block. Please refer to `Aggregated
    Residual Transformations for Deep Neural Networks
    <https://arxiv.org/pdf/1611.05431.pdf>`_ and `Squeeze-and-Excitation
    Networks <https://arxiv.org/pdf/1709.01507.pdf>`_ for detailed
    description of network architecture.

    Similar to :class:`chainercv.links.model.resnet.ResNet`, ImageNet
    pretrained weights are downloaded when :obj:`pretrained_model` argument
    is :obj:`imagenet`, originally distributed at `the Github repository by
    one of the paper authors of SENet <https://github.com/hujie-frank/SENet>`_.

    .. seealso::
        :class:`chainercv.links.model.resnet.ResNet`
        :class:`chainercv.links.model.senet.SEResNet`
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
    }

    preset_params = {'imagenet': {'n_class': 1000, 'mean': _imagenet_mean}}

    _models = {
        50: {
            'imagenet': {
                'param': preset_params['imagenet'],
                'overwritable': {'mean'},
                'url': 'https://chainercv-models.preferred.jp/'
                'se_resnext50_imagenet_converted_2018_06_28.npz'
            },
        },
        101: {
            'imagenet': {
                'param': preset_params['imagenet'],
                'overwritable': {'mean'},
                'url': 'https://chainercv-models.preferred.jp/'
                'se_resnext101_imagenet_converted_2018_06_28.npz'
            },
        },
    }

    def __init__(self, n_layer,
                 n_class=None,
                 pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        blocks = self._blocks[n_layer]

        param, path = utils.prepare_model_param(
            locals(), self._models[n_layer])
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
            'groups': 32, 'initialW': initialW, 'stride_first': False,
            'add_seblock': True}

        super(SEResNeXt, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 7, 2, 3, nobias=True,
                                       initialW=initialW)
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.res2 = ResBlock(blocks[0], None, 128, 256, 1, **kwargs)
            self.res3 = ResBlock(blocks[1], None, 256, 512, 2, **kwargs)
            self.res4 = ResBlock(blocks[2], None, 512, 1024, 2, **kwargs)
            self.res5 = ResBlock(blocks[3], None, 1024, 2048, 2, **kwargs)
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(None, param['n_class'], **fc_kwargs)
            self.prob = F.softmax

        if path:
            chainer.serializers.load_npz(path, self)


class SEResNeXt50(SEResNeXt):

    """SE-ResNeXt-50 Network

    Please consult the documentation for :class:`SEResNeXt`.

    .. seealso::
        :class:`chainercv.links.model.senet.SEResNeXt`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNeXt50, self).__init__(
            50, n_class, pretrained_model,
            mean, initialW, fc_kwargs)


class SEResNeXt101(SEResNeXt):

    """SE-ResNeXt-101 Network

    Please consult the documentation for :class:`SEResNeXt`.

    .. seealso::
        :class:`chainercv.links.model.senet.SEResNeXt`

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNeXt101, self).__init__(
            101, n_class, pretrained_model,
            mean, initialW, fc_kwargs)
