from __future__ import division

import collections

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links as L

from chainercv.transforms import center_crop
from chainercv.transforms import scale
from chainercv.transforms import ten_crop

from chainercv.utils import download_model


# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


class VGG16Layers(chainer.Chain):

    """VGG16 Network for classification and feature extraction.

    This model is a feature extraction link.
    The network can choose to output features from set of all
    intermediate and final features produced by the original architecture.

    When :obj:`pretrained_model` is thepath of a pre-trained chainer model
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
        feature (str or iterable of strings): The name of the feature to output
            with :meth:`__call__` and :meth:`predict`.
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
                 features='prob', initialW=None, initial_bias=None,
                 mean=_imagenet_mean, do_ten_crop=False):
        if isinstance(features, (list, tuple)):
            return_tuple = True
        else:
            return_tuple = False
            features = [features]

        self._return_tuple = return_tuple
        self._features = features
        self.mean = mean
        self.do_ten_crop = do_ten_crop

        if n_class is None:
            if (pretrained_model is None and
                    all([feature not in ['fc8', 'prob']
                         for feature in features])):
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

        super(VGG16Layers, self).__init__()

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
        with self.init_scope():
            for name, link_gen in link_generators.items():
                if name in self.functions:
                    setattr(self, name, link_gen())

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    @property
    def functions(self):
        def _getattr(name):
            return getattr(self, name, None)

        funcs = collections.OrderedDict([
            ('conv1_1', [_getattr('conv1_1'), F.relu]),
            ('conv1_2', [_getattr('conv1_2'), F.relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [_getattr('conv2_1'), F.relu]),
            ('conv2_2', [_getattr('conv2_2'), F.relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [_getattr('conv3_1'), F.relu]),
            ('conv3_2', [_getattr('conv3_2'), F.relu]),
            ('conv3_3', [_getattr('conv3_3'), F.relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [_getattr('conv4_1'), F.relu]),
            ('conv4_2', [_getattr('conv4_2'), F.relu]),
            ('conv4_3', [_getattr('conv4_3'), F.relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [_getattr('conv5_1'), F.relu]),
            ('conv5_2', [_getattr('conv5_2'), F.relu]),
            ('conv5_3', [_getattr('conv5_3'), F.relu]),
            ('pool5', [_max_pooling_2d]),
            ('fc6', [_getattr('fc6'), F.relu, F.dropout]),
            ('fc7', [_getattr('fc7'), F.relu, F.dropout]),
            ('fc8', [_getattr('fc8')]),
            ('prob', [F.softmax]),
        ])
        for name in self._features:
            if name not in funcs:
                raise ValueError('Elements of `features` shuold be one of '
                                 '{}.'.format(funcs.keys()))

        # Remove all functions that are not necessary.
        pop_funcs = False
        features = list(self._features)
        for name in funcs.keys():
            if pop_funcs:
                funcs.pop(name)

            if name in features:
                features.remove(name)
            if len(features) == 0:
                pop_funcs = True

        return funcs

    def __call__(self, x):
        """Fowrard VGG16.

        Args:
            x (~chainer.Variable): Batch of image variables.

        Returns:
            ~chainer.Variable:
            A batch of features. It is selected by :obj:`self._feature`.

        """
        activations = {}
        h = x
        for name, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            if name in self._features:
                activations[name] = h

        if self._return_tuple:
            activations = tuple(
                [activations[name] for name in activations.keys()])
        else:
            activations = activations.values()[0]
        return activations

    def _prepare(self, img):
        """Transform an image to the input for VGG network.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        img = scale(img, size=256)
        img = img - self.mean

        return img

    def _average_ten_crop(self, y):
        xp = chainer.cuda.get_array_module(y)
        n = y.shape[0] // 10
        y_shape = y.shape[1:]
        y = y.reshape((n, 10) + y_shape)
        y = xp.sum(y, axis=1) / 10
        return y

    def predict(self, imgs):
        """Predict features from images.

        When :obj:`self.do_ten_crop == True`, this extracts features from
        patches that are ten-cropped from images.
        Otherwise, this extracts features from center-crop of the images.

        When using patches from ten-crop, the features for each crop
        is averaged to compute one feature.
        Ten-crop mode is only supported for calculation of features
        :math:`fc6, fc7, fc8, prob`.

        Given :math:`N` input images, this outputs a batched array with
        batchsize :math:`N`.

        Args:
            imgs (iterable of numpy.ndarray): Array-images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
            numpy.ndarray:
            A batch of features. It is selected by :obj:`self._feature`.

        """
        if (self.do_ten_crop and
                any([feature not in ['fc6', 'fc7', 'fc8', 'prob']
                     for feature in self._features])):
            raise ValueError

        imgs = [self._prepare(img) for img in imgs]
        if self.do_ten_crop:
            imgs = [ten_crop(img, (224, 224)) for img in imgs]
        else:
            imgs = [center_crop(img, (224, 224)) for img in imgs]
        imgs = self.xp.asarray(imgs).reshape(-1, 3, 224, 224)

        with chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            activations = self(imgs)

        if isinstance(activations, tuple):
            output = []
            for activation in activations:
                activation = activation.data
                if self.do_ten_crop:
                    activation = self._average_ten_crop(activation)
                output.append(cuda.to_cpu(activation))
            output = tuple(output)
        else:
            output = cuda.to_cpu(activations.data)
            if self.do_ten_crop:
                output = self._average_ten_crop(output)

        return output


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)
