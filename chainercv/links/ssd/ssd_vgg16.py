import numpy as np
import os
import six

import chainer
from chainer.dataset.download import get_dataset_directory
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links.ssd import Normalize
from chainercv.links.ssd import SSD
from chainercv.utils import download


_imagenet_mean = (104, 117, 123)


class VGG16Extractor(chainer.Chain):

    def __init__(self, **links):
        super(VGG16Extractor, self).__init__(
            conv1_1=L.Convolution2D(None, 64, 3, pad=1),
            conv1_2=L.Convolution2D(None, 64, 3, pad=1),

            conv2_1=L.Convolution2D(None, 128, 3, pad=1),
            conv2_2=L.Convolution2D(None, 128, 3, pad=1),

            conv3_1=L.Convolution2D(None, 256, 3, pad=1),
            conv3_2=L.Convolution2D(None, 256, 3, pad=1),
            conv3_3=L.Convolution2D(None, 256, 3, pad=1),

            conv4_1=L.Convolution2D(None, 512, 3, pad=1),
            conv4_2=L.Convolution2D(None, 512, 3, pad=1),
            conv4_3=L.Convolution2D(None, 512, 3, pad=1),
            norm4=Normalize(512, initial=initializers.Constant(20)),

            conv5_1=L.DilatedConvolution2D(None, 512, 3, pad=1),
            conv5_2=L.DilatedConvolution2D(None, 512, 3, pad=1),
            conv5_3=L.DilatedConvolution2D(None, 512, 3, pad=1),

            conv6=L.DilatedConvolution2D(None, 1024, 3, pad=6, dilate=6),
            conv7=L.Convolution2D(None, 1024, 1),
        )
        for name, link in six.iteritems(links):
            self.add_link(name, link)

    def __call__(self, x):
        ys = list()

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        ys.append(self.norm4(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        ys.append(h)

        return ys


class VGG16Extractor300(VGG16Extractor):
    insize = 300
    grids = (38, 19, 10, 5, 3, 1)

    def __init__(self):
        init = {
            'initialW': initializers.GlorotUniform(),
            'initial_bias': initializers.Zero(),
        }
        super(VGG16Extractor300, self).__init__(
            conv8_1=L.Convolution2D(None, 256, 1, **init),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1, **init),

            conv9_1=L.Convolution2D(None, 128, 1, **init),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv10_1=L.Convolution2D(None, 128, 1, **init),
            conv10_2=L.Convolution2D(None, 256, 3, **init),

            conv11_1=L.Convolution2D(None, 128, 1, **init),
            conv11_2=L.Convolution2D(None, 256, 3, **init),
        )

    def __call__(self, x):
        """Compute feature maps from a batch of images.

        This method extracts feature maps from
        :obj:`conv4_3`, :obj:`conv7`, :obj:`conv8_2`,
        :obj:`conv9_2`, :obj:`conv10_2`, and :obj:`conv11_2`.

        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized and rescaled by :meth:`prepare`.

        Returns:
            list of Variable:
            Each variable contains a feature map.
        """

        ys = super(VGG16Extractor300, self).__call__(x)
        for i in range(8, 11 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys


class VGG16Extractor512(VGG16Extractor):
    insize = 512
    grids = (64, 32, 16, 8, 4, 2, 1)

    def __init__(self):
        init = {
            'initialW': initializers.GlorotUniform(),
            'initial_bias': initializers.Zero(),
        }
        super(VGG16Extractor512, self).__init__(
            conv8_1=L.Convolution2D(None, 256, 1, **init),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1, **init),

            conv9_1=L.Convolution2D(None, 128, 1, **init),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv10_1=L.Convolution2D(None, 128, 1, **init),
            conv10_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv11_1=L.Convolution2D(None, 128, 1, **init),
            conv11_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv12_1=L.Convolution2D(None, 128, 1, **init),
            conv12_2=L.Convolution2D(None, 256, 4,  pad=1, **init),
        )

    def __call__(self, x):
        """Compute feature maps from a batch of images.

        This method extracts feature maps from
        :obj:`conv4_3`, :obj:`conv7`, :obj:`conv8_2`,
        :obj:`conv9_2`, :obj:`conv10_2`, :obj:`conv11_2`, and :obj:`conv12_2`.

        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized and rescaled by :meth:`prepare`.

        Returns:
            list of Variable:
            Each variable contains a feature map.
        """

        ys = super(VGG16Extractor512, self).__call__(x)
        for i in range(8, 12 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys


class SSD300(SSD):
    """Single Shot Multibox Detector.

    This is a model of Single Shot Multibox Detector.
    This model is based on VGG-16 and takes 300x300 images as inputs.

    This model is proposed in [#]_.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (str): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on Pascal VOC 2007 and \
                2012. The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    """

    mean = _imagenet_mean
    _models = {
        'voc0712': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.1/ssd300_voc0712.npz'
        }
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        if pretrained_model in self._models:
            model = self._models[pretrained_model]
            if n_fg_class and not n_fg_class == model['n_fg_class']:
                raise ValueError('n_fg_class mismatch')
            n_fg_class = model['n_fg_class']

            root = get_dataset_directory('pfnet/chainercv/models')
            basename = os.path.basename(model['url'])
            path = os.path.join(root, basename)
            if not os.path.exists(path):
                download_file = download.cached_download(model['url'])
                os.rename(download_file, path)
        elif pretrained_model:
            path = pretrained_model
        else:
            path = None

        super(SSD300, self).__init__(
            n_fg_class,
            extractor=VGG16Extractor300(),
            aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
            steps=[s / 300 for s in (8, 16, 32, 64, 100, 300)],
            sizes=[s / 300 for s in (30, 60, 111, 162, 213, 264, 315)])

        if path:
            chainer.serializers.load_npz(path, self)


class SSD512(SSD):
    """Single Shot Multibox Detector.

    This is a model of Single Shot Multibox Detector.
    This model is based on VGG-16 and takes 512x512 images as inputs.

    This model is proposed in [#]_.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (str): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on Pascal VOC 2007 and \
                2012. The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    """

    mean = _imagenet_mean
    _models = {
        'voc0712': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.1/ssd512_voc0712.npz'
        }
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        if pretrained_model in self._models:
            model = self._models[pretrained_model]
            if n_fg_class and not n_fg_class == model['n_fg_class']:
                raise ValueError('n_fg_class mismatch')
            n_fg_class = model['n_fg_class']

            root = get_dataset_directory('pfnet/chainercv/models')
            basename = os.path.basename(model['url'])
            path = os.path.join(root, basename)
            if not os.path.exists(path):
                download_file = download.cached_download(model['url'])
                os.rename(download_file, path)
        elif pretrained_model:
            path = pretrained_model
        else:
            path = None

        super(SSD512, self).__init__(
            n_fg_class,
            extractor=VGG16Extractor512(),
            aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,)),
            steps=[s / 512 for s in (8, 16, 32, 64, 128, 256, 512)],
            sizes=[s / 512 for s in
                   (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)])

        if path:
            chainer.serializers.load_npz(path, self)
