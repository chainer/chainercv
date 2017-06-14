from __future__ import division

import collections

import numpy as np

import chainer
import chainer.functions as F
from chainer import cuda
from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links as L

from chainercv.transforms import center_crop
from chainercv.transforms import scale
from chainercv.transforms import ten_crop


# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


class VGG16Layers(chainer.Chain):

    def __init__(self, pretrained_model='auto', feature='prob',
                 initialW=None, initial_bias=None,
                 mean=_imagenet_mean, do_ten_crop=False):
        self.mean = mean
        self.do_ten_crop = do_ten_crop
        if do_ten_crop and feature not in ['fc6', 'fc7', 'fc8', 'prob']:
            raise ValueError

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            if initialW is None:
                initialW = constant.Zero()
            if initial_bias is None:
                initial_bias = constant.Zero()
        else:
            # employ default initializers used in the original paper
            if initialW is None:
                initialW = normal.Normal(0.01)
            if initial_bias is None:
                initial_bias = constant.Zero()

        kwargs = {'initialW': initialW, 'initial_bias': initial_bias}

        super(VGG16Layers, self).__init__()

        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.fc6 = L.Linear(512 * 7 * 7, 4096, **kwargs)
            self.fc7 = L.Linear(4096, 4096, **kwargs)
            self.fc8 = L.Linear(4096, 1000, **kwargs)
        self.feature = feature

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

        # Links can be safely removed because parameters in these links
        # are guaranteed to not be in a computational graph.
        names = [child.name for child in self.children()]
        functions = self.functions
        for name in names:
            if name not in functions:
                delattr(self, name)
                # Since self.functions access self.name, it needs a value.
                setattr(self, name, None)

    @property
    def functions(self):
        default_funcs = collections.OrderedDict([
            ('conv1_1', [self.conv1_1, F.relu]),
            ('conv1_2', [self.conv1_2, F.relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, F.relu]),
            ('conv2_2', [self.conv2_2, F.relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, F.relu]),
            ('conv3_2', [self.conv3_2, F.relu]),
            ('conv3_3', [self.conv3_3, F.relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, F.relu]),
            ('conv4_2', [self.conv4_2, F.relu]),
            ('conv4_3', [self.conv4_3, F.relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, F.relu]),
            ('conv5_2', [self.conv5_2, F.relu]),
            ('conv5_3', [self.conv5_3, F.relu]),
            ('pool5', [_max_pooling_2d]),
            ('fc6', [self.fc6, F.relu, F.dropout]),
            ('fc7', [self.fc7, F.relu, F.dropout]),
            ('fc8', [self.fc8]),
            ('prob', [F.softmax]),
        ])
        if self.feature not in default_funcs:
            raise ValueError('`feature` shuold be one of the keys of '
                             'VGG16Layers.functions.')
        pop_funcs = False
        for name in default_funcs.keys():
            if pop_funcs:
                default_funcs.pop(name)

            if name == self.feature:
                pop_funcs = True
        return default_funcs

    def __call__(self, x):
        h = x
        for funcs in self.functions.values():
            for func in funcs:
                h = func(h)
        return h

    def _prepare(self, img):
        """Transform an image to the input for VGG network.

        Args:
            img (numpy.ndarray)

        Returns:
            numpy.ndarray: The transformed image.

        """

        img = scale(img, size=256)
        img = img - self.mean

        return img

    def predict(self, imgs):
        """Compute class probabilities of given images.

        Args:
            imgs (iterable of numpy.ndarray): Array-images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.
            do_ten_crop (bool): If :obj:`True`, it averages results across
                center, corners, and mirrors. Otherwise, it uses only the
                center.

        Returns:
            numpy.ndarray:
            A batch of arrays containing class-probabilities.

        """
        imgs = [self._prepare(img) for img in imgs]
        if self.do_ten_crop:
            imgs = [ten_crop(img, (224, 224)) for img in imgs]
        else:
            imgs = [center_crop(img, (224, 224)) for img in imgs]
        imgs = self.xp.asarray(imgs).reshape(-1, 3, 224, 224)

        with chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            y = self(imgs).data

            if self.do_ten_crop:
                n = y.shape[0] // 10
                y_shape = y.shape[1:]
                y = y.reshape((n, 10) + y_shape)
                y = self.xp.sum(y, axis=1) / 10
        return cuda.to_cpu(y)


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)
