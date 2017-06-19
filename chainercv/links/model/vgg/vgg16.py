from __future__ import division

import collections
from itertools import islice

import chainer
import chainer.functions as F
from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links as L

from chainercv.utils import download_model

from chainercv.links.model.sequential_chain import SequentialChain


class VGG16(SequentialChain):

    """VGG16 Network for classification and feature extraction.

    This model is a feature extraction link.
    The network can choose to output features from set of all
    intermediate and final features produced by the original architecture.
    The output features can be an array or tuple of arrays.
    When :obj:`features` is an iterable of strings, outputs will be tuple.
    When :obj:`features` is a string, output will be an array.

    Examples:

        >>> model = VGG16(features='conv5_3')
        # This is an activation of conv5_3 layer.
        >>> feat = model(imgs)

        >>> model = VGG16(features=['conv5_3', 'fc6'])
        >>> # These are activations of conv5_3 and fc6 layers respectively.
        >>> feat1, feat2 = model(imgs)

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
        n_class (int): The dimension of the output of fc8.
        features (str or iterable of strings): The names of the feature to
            output with :meth:`__call__` and :meth:`predict`.
        initialW (callable): Initializer for the weights.
        initial_bias (callable): Initializer for the biases.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :meth:`_prepare`.
        do_ten_crop (bool): If :obj:`True`, it averages results across
            center, corners, and mirrors in :meth:`predict`. Otherwise, it uses
            only the center. The default value is :obj:`False`.

    """

    _models = {
        'imagenet': {
            'n_class': 1000,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.3/vgg16_imagenet_convert_2017_06_15.npz'
        }
    }

    def __init__(self, pretrained_model=None, n_class=None,
                 feature_names='prob', initialW=None, initial_bias=None):
        if n_class is None:
            if (pretrained_model is None and
                    all([feature not in ['fc8', 'prob']
                         for feature in feature_names])):
                # fc8 layer is not used in this case.
                pass
            elif pretrained_model not in self._models:
                raise ValueError(
                    'The n_class needs to be supplied as an argument.')
            else:
                n_class = self._models[pretrained_model]['n_class']

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            if initialW is None:
                initialW = constant.Zero()
            if initial_bias is None:
                initial_bias = constant.Zero()
        else:
            # Employ default initializers used in the original paper.
            if initialW is None:
                initialW = normal.Normal(0.01)
            if initial_bias is None:
                initial_bias = constant.Zero()
        kwargs = {'initialW': initialW, 'initial_bias': initial_bias}

        link_generators = {
            'conv1_1': lambda: L.Convolution2D(3, 64, 3, 1, 1, **kwargs),
            'conv1_2': lambda: L.Convolution2D(64, 64, 3, 1, 1, **kwargs),
            'conv2_1': lambda: L.Convolution2D(64, 128, 3, 1, 1, **kwargs),
            'conv2_2': lambda: L.Convolution2D(128, 128, 3, 1, 1, **kwargs),
            'conv3_1': lambda: L.Convolution2D(128, 256, 3, 1, 1, **kwargs),
            'conv3_2': lambda: L.Convolution2D(256, 256, 3, 1, 1, **kwargs),
            'conv3_3': lambda: L.Convolution2D(256, 256, 3, 1, 1, **kwargs),
            'conv4_1': lambda: L.Convolution2D(256, 512, 3, 1, 1, **kwargs),
            'conv4_2': lambda: L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            'conv4_3': lambda: L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            'conv5_1': lambda: L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            'conv5_2': lambda: L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            'conv5_3': lambda: L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            'fc6': lambda: L.Linear(512 * 7 * 7, 4096, **kwargs),
            'fc7': lambda: L.Linear(4096, 4096, **kwargs),
            'fc8': lambda: L.Linear(4096, n_class, **kwargs)
        }

        # None will be initialized according to link_generators once
        # the functions to use are selected.
        functions = collections.OrderedDict([
            ('conv1_1', None),
            ('conv1_1_relu', F.relu),
            ('conv1_2', None),
            ('conv1_2_relu', F.relu),
            ('pool1', _max_pooling_2d),
            ('conv2_1', None),
            ('conv2_1_relu', F.relu),
            ('conv2_2', None),
            ('conv2_2_relu', F.relu),
            ('pool2', _max_pooling_2d),
            ('conv3_1', None),
            ('conv3_1_relu', F.relu),
            ('conv3_2', None),
            ('conv3_2_relu', F.relu),
            ('conv3_3', None),
            ('conv3_3_relu', F.relu),
            ('pool3', _max_pooling_2d),
            ('conv4_1', None),
            ('conv4_1_relu', F.relu),
            ('conv4_2', None),
            ('conv4_2_relu', F.relu),
            ('conv4_3', None),
            ('conv4_3_relu', F.relu),
            ('pool4', _max_pooling_2d),
            ('conv5_1', None),
            ('conv5_1_relu', F.relu),
            ('conv5_2', None),
            ('conv5_2_relu', F.relu),
            ('conv5_3', None),
            ('conv5_3_relu', F.relu),
            ('pool5', _max_pooling_2d),
            ('fc6', None),
            ('fc6_relu', F.relu),
            ('fc6_dropout', F.dropout),
            ('fc7', None),
            ('fc7_relu', F.relu),
            ('fc7_dropout', F.dropout),
            ('fc8', None),
            ('prob', F.softmax)
            ])
        functions = _choose_necessary_functions(functions, feature_names)
        # Instantiate uninitialized links.
        for name in list(functions.keys()):
            if name in link_generators:
                functions[name] = link_generators[name]()
        super(VGG16, self).__init__(functions, feature_names)

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)


def _choose_necessary_functions(functions, feature_names):
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    last_index = max([list(functions.keys()).index(name) for
                      name in feature_names])
    # Equivalent to `functions = functions[:last_index + 1]`.
    functions = collections.OrderedDict(
        islice(functions.items(), None, last_index + 1))
    return functions
