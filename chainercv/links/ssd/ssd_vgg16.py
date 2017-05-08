import numpy as np
import six

import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links.ssd import Normalize
from chainercv.links.ssd import SSD
from chainercv import transforms


class SSDVGG16(SSD):
    mean = (104, 117, 123)

    def __init__(self, n_class, **links):
        super(SSDVGG16, self).__init__(
            n_class,

            conv1_1=L.Convolution2D(None, 64, 3, pad=1, **self.conv_init),
            conv1_2=L.Convolution2D(None, 64, 3, pad=1, **self.conv_init),

            conv2_1=L.Convolution2D(None, 128, 3, pad=1, **self.conv_init),
            conv2_2=L.Convolution2D(None, 128, 3, pad=1, **self.conv_init),

            conv3_1=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),
            conv3_2=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),
            conv3_3=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),

            conv4_1=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            conv4_2=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            conv4_3=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            norm4=Normalize(512, initial=initializers.Constant(20)),

            conv5_1=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),
            conv5_2=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),
            conv5_3=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),

            conv6=L.DilatedConvolution2D(
                None, 1024, 3, pad=6, dilate=6, **self.conv_init),
            conv7=L.Convolution2D(None, 1024, 1, **self.conv_init),
        )
        for name, link in six.iteritems(links):
            self.add_link(name, link)

    def features(self, x):
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

    def prepare(self, img):
        img = transforms.resize(img, (self.insize, self.insize))
        img -= np.array(self.mean)[:, np.newaxis, np.newaxis]
        return img


class SSD300(SSDVGG16):
    """Single Shot Multibox Detector.

    This is a model of Single Shot Multibox Detector.
    This model is based on VGG-16 and takes 300x300 images as inputs.

    This model is proposed in [1].

    [1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
    SSD: Single Shot MultiBox Detector. ECCV 2016.
    """

    insize = 300
    grids = (38, 19, 10, 5, 3, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)]

    def __init__(self, n_class):
        super(SSD300, self).__init__(
            n_class,

            conv8_1=L.Convolution2D(None, 256, 1, **self.conv_init),
            conv8_2=L.Convolution2D(
                None, 512, 3, stride=2, pad=1, **self.conv_init),

            conv9_1=L.Convolution2D(None, 128, 1, **self.conv_init),
            conv9_2=L.Convolution2D(
                None, 256, 3, stride=2, pad=1, **self.conv_init),

            conv10_1=L.Convolution2D(None, 128, 1, **self.conv_init),
            conv10_2=L.Convolution2D(None, 256, 3, **self.conv_init),

            conv11_1=L.Convolution2D(None, 128, 1, **self.conv_init),
            conv11_2=L.Convolution2D(None, 256, 3, **self.conv_init),
        )

    def features(self, x):
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

        ys = super(SSD300, self).features(x)
        for i in range(8, 11 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys


class SSD512(SSDVGG16):
    """Single Shot Multibox Detector.

    This is a model of Single Shot Multibox Detector.
    This model is based on VGG-16 and takes 512x512 images as inputs.

    This model is proposed in [1].

    [1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
    SSD: Single Shot MultiBox Detector. ECCV 2016.
    """

    insize = 512
    grids = (64, 32, 16, 8, 4, 2, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2, ))
    steps = [s / 512 for s in (8, 16, 32, 64, 128, 256, 512)]
    sizes = [s / 512 for s in
             (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)]

    def __init__(self, n_class):
        super(SSD512, self).__init__(
            n_class,

            conv8_1=L.Convolution2D(None, 256, 1, **self.conv_init),
            conv8_2=L.Convolution2D(
                None, 512, 3, stride=2, pad=1, **self.conv_init),

            conv9_1=L.Convolution2D(None, 128, 1, **self.conv_init),
            conv9_2=L.Convolution2D(
                None, 256, 3, stride=2, pad=1, **self.conv_init),

            conv10_1=L.Convolution2D(None, 128, 1, **self.conv_init),
            conv10_2=L.Convolution2D(
                None, 256, 3, stride=2, pad=1, **self.conv_init),

            conv11_1=L.Convolution2D(None, 128, 1, **self.conv_init),
            conv11_2=L.Convolution2D(
                None, 256, 3, stride=2, pad=1, **self.conv_init),

            conv12_1=L.Convolution2D(None, 128, 1, **self.conv_init),
            conv12_2=L.Convolution2D(
                None, 256, 4,  pad=1, **self.conv_init),
        )

    def features(self, x):
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

        ys = super(SSD512, self).features(x)
        for i in range(8, 12 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys
