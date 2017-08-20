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

from chainercv.utils import download_model

from chainercv.links.connection.conv_2d_activ import Conv2DActiv
from chainercv.links.model.sequential_feature_extractor import \
    SequentialFeatureExtractor


# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


class VGG16(SequentialFeatureExtractor):

    """VGG16 Network for classification and feature extraction.

    This is a feature extraction model.
    The network can choose to output features from set of all
    intermediate features.
    The value of :obj:`VGG16.feature_names` selects the features that are going
    to be collected by :meth:`__call__`.
    :obj:`self.all_feature_names` is the list of the names of features
    that can be collected.

    Examples:

        >>> model = VGG16()
        # By default, VGG16.__call__ returns a probability score.
        >>> prob = model(imgs)

        >>> model.feature_names = 'conv5_3'
        # This is feature conv5_3.
        >>> feat5_3 = model(imgs)

        >>> model.feature_names = ['conv5_3', 'fc6']
        >>> # These are features conv5_3 and fc6.
        >>> feat5_3, feat6 = model(imgs)

    .. seealso::
        :class:`chainercv.links.model.SequentialFeatureExtractor`

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
        pretrained_model (str): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        n_class (int): The number of classes.
        mean (numpy.ndarray): A mean value. If :obj:`None` and
            a supported pretrained model is used,
            the mean value used to train the pretrained model will be used.
        initialW (callable): Initializer for the weights.
        initial_bias (callable): Initializer for the biases.

    """

    _models = {
        'imagenet': {
            'n_class': 1000,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.4/vgg16_imagenet_convert_2017_07_18.npz',
            'mean': _imagenet_mean
        }
    }

    def __init__(self,
                 pretrained_model=None, n_class=None, mean=None,
                 initialW=None, initial_bias=None):
        if n_class is None:
            if pretrained_model in self._models:
                n_class = self._models[pretrained_model]['n_class']
            else:
                n_class = 1000

        if mean is None:
            if pretrained_model in self._models:
                mean = self._models[pretrained_model]['mean']
            else:
                mean = _imagenet_mean
        self.mean = mean

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
            self.fc8 = Linear(None, n_class, **kwargs)
            self.prob = softmax

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=2)
