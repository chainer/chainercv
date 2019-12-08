from __future__ import division

import numpy as np

import chainer
from chainer.functions import dropout
from chainer.functions import max_pooling_2d
from chainer.functions import relu
from chainer.functions import softmax
from chainer.initializers import constant
from chainer.initializers import normal

from chainer.links import Linear

from chainercv.links.connection.conv_2d_activ import Conv2DActiv
from chainercv.links.model.pickable_sequential_chain import \
    PickableSequentialChain
from chainercv import utils


# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


class VGG16(PickableSequentialChain):

    """VGG-16 Network.

    This is a pickable sequential link.
    The network can choose output layers from set of all
    intermediate layers.
    The attribute :obj:`pick` is the names of the layers that are going
    to be picked by :meth:`forward`.
    The attribute :obj:`layer_names` is the names of all layers
    that can be picked.

    Examples:

        >>> model = VGG16()
        # By default, forward returns a probability score (after Softmax).
        >>> prob = model(imgs)
        >>> model.pick = 'conv5_3'
        # This is layer conv5_3 (after ReLU).
        >>> conv5_3 = model(imgs)
        >>> model.pick = ['conv5_3', 'fc6']
        >>> # These are layers conv5_3 (after ReLU) and fc6 (before ReLU).
        >>> conv5_3, fc6 = model(imgs)

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

    Args:
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
        initialW (callable): Initializer for the weights.
        initial_bias (callable): Initializer for the biases.

    """

    _models = {
        'imagenet': {
            'param': {'n_class': 1000, 'mean': _imagenet_mean},
            'overwritable': ('mean',),
            'url': 'https://chainercv-models.preferred.jp/'
            'vgg16_imagenet_converted_2017_07_18.npz'
        }
    }

    def __init__(self,
                 n_class=None, pretrained_model=None, mean=None,
                 initialW=None, initial_bias=None):
        param, path = utils.prepare_pretrained_model(
            {'n_class': n_class, 'mean': mean},
            pretrained_model, self._models,
            {'n_class': 1000, 'mean': _imagenet_mean})
        self.mean = param['mean']

        if initialW is None:
            # Employ default initializers used in the original paper.
            initialW = normal.Normal(0.01)
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            initialW = constant.Zero()
        kwargs = {'initialW': initialW, 'initial_bias': initial_bias}

        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = Conv2DActiv(None, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = Conv2DActiv(None, 64, 3, 1, 1, **kwargs)
            self.pool1 = _max_pooling_2d
            self.conv2_1 = Conv2DActiv(None, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = Conv2DActiv(None, 128, 3, 1, 1, **kwargs)
            self.pool2 = _max_pooling_2d
            self.conv3_1 = Conv2DActiv(None, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Conv2DActiv(None, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = Conv2DActiv(None, 256, 3, 1, 1, **kwargs)
            self.pool3 = _max_pooling_2d
            self.conv4_1 = Conv2DActiv(None, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Conv2DActiv(None, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = Conv2DActiv(None, 512, 3, 1, 1, **kwargs)
            self.pool4 = _max_pooling_2d
            self.conv5_1 = Conv2DActiv(None, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Conv2DActiv(None, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = Conv2DActiv(None, 512, 3, 1, 1, **kwargs)
            self.pool5 = _max_pooling_2d
            self.fc6 = Linear(None, 4096, **kwargs)
            self.fc6_relu = relu
            self.fc6_dropout = dropout
            self.fc7 = Linear(None, 4096, **kwargs)
            self.fc7_relu = relu
            self.fc7_dropout = dropout
            self.fc8 = Linear(None, param['n_class'], **kwargs)
            self.prob = softmax

        if path:
            chainer.serializers.load_npz(path, self)


def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=2)
