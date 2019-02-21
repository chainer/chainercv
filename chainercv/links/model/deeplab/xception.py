from __future__ import division

import chainer
import chainer.functions as F

from chainercv.links import Conv2DBNActiv
from chainercv.links import SeparableConv2DBNActiv

import numpy as np


class XceptionBlock(chainer.Chain):

    """A building block for Xceptions.

    Not only final outputs, this block also returns unactivated outputs
    of second separable convolution.

    Args:
        in_channels (int): The number of channels of the input array.
        depthlist (tuple of ints): Tuple of integers which defines
            number of channels of intermediate arrays. The length of
            this tuple must be 3.
        stride (int or tuple of ints): Stride of filter application.
        dilate (int or tuple of ints): Dilation factor of filter applications.
            :obj:`dilate=d` and :obj:`dilate=(d, d)` are equivalent.
        skip_type (string): the type of skip connection. If :obj:`sum`,
            original input is summed to output of network directly.
            When :obj:`conv`, convolution layer is applied before summation.
            When :obj:`none`, skip connection is not used.
            The default value is :obj:`conv`.
        activ_first (boolean): If :obj:`True`, activation function is
            applied first in this block.
            The default value is :obj:`True`
        bn_kwargs (dict):  Keywod arguments passed to initialize the batch
            normalization layers of :class:`chainercv.links.Conv2DBNActiv` and
            :class:`chainercv.links.SeparableConv2DBNActiv`.

    """

    def __init__(self, in_channels, depthlist, stride=1, dilate=1,
                 skip_type='conv', activ_first=True, bn_kwargs={},
                 dw_activ_list=[None, None, None],
                 pw_activ_list=[F.relu, F.relu, None]):
        super(XceptionBlock, self).__init__()
        self.skip_type = skip_type
        self.activ_first = activ_first
        self.separable2_activ = pw_activ_list[1]

        with self.init_scope():
            self.separable1 = SeparableConv2DBNActiv(
                in_channels, depthlist[0], 3, 1,
                dilate, dilate, nobias=True, bn_kwargs=bn_kwargs,
                dw_activ=dw_activ_list[0], pw_activ=pw_activ_list[0])
            self.separable2 = SeparableConv2DBNActiv(
                depthlist[0], depthlist[1], 3, 1,
                dilate, dilate, nobias=True, bn_kwargs=bn_kwargs,
                dw_activ=dw_activ_list[1], pw_activ=F.identity)
            self.separable3 = SeparableConv2DBNActiv(
                depthlist[1], depthlist[2], 3, stride,
                dilate, dilate, nobias=True, bn_kwargs=bn_kwargs,
                dw_activ=dw_activ_list[2], pw_activ=pw_activ_list[2])
            if skip_type == 'conv':
                self.conv = Conv2DBNActiv(
                    in_channels, depthlist[2], 1, activ=F.identity,
                    nobias=True, stride=stride, bn_kwargs=bn_kwargs)

    def __call__(self, x):
        if self.activ_first:
            h = F.relu(x)
        else:
            h = x

        h = self.separable1(h)
        h = self.separable2(h)
        separable2 = h
        h = self.separable2_activ(h)
        h = self.separable3(h)

        if self.skip_type == 'conv':
            skip = self.conv(x)
            h = h + skip
        elif self.skip_type == 'sum':
            h = h + x
        elif self.skip_type == 'none':
            pass

        if not self.activ_first:
            h = F.relu(h)

        return h, separable2


class Xception65(chainer.Chain):

    """Xception65 for backbone network of DeepLab v3+.

    Unlike original Xception65, this follows implementation in deeplab v3
    (https://github.com/tensorflow/models/tree/master/research/deeplab).
    This returns lowlevel feature (an output of second convolution in second
    block in entryflow) and highlevel feature (an output before final average
    pooling in original).

    Args:
        bn_kwargs (dict):  Keywod arguments passed to initialize the batch
            normalization layers of :class:`chainercv.links.Conv2DBNActiv` and
            :class:`chainercv.links.SeparableConv2DBNActiv`.

    """

    mean = np.array([127.5, 127.5, 127.5],
                    dtype=np.float32)[:, np.newaxis, np.newaxis]

    def __init__(self, bn_kwargs={}):
        super(Xception65, self).__init__()

        with self.init_scope():
            self.entryflow_conv1 = Conv2DBNActiv(
                3, 32, 3, 2, 1, bn_kwargs=bn_kwargs)
            self.entryflow_conv2 = Conv2DBNActiv(
                32, 64, 3, 1, 1, bn_kwargs=bn_kwargs)
            self.entryflow_block1 = XceptionBlock(
                64, [128, 128, 128], stride=2,
                skip_type='conv', bn_kwargs=bn_kwargs)
            self.entryflow_block2 = XceptionBlock(
                128, [256, 256, 256], stride=2,
                skip_type='conv', bn_kwargs=bn_kwargs)
            self.entryflow_block3 = XceptionBlock(
                256, [728, 728, 728], stride=1,
                skip_type='conv', bn_kwargs=bn_kwargs)

            for i in range(1, 17):
                block = XceptionBlock(
                    728, [728, 728, 728], stride=1, dilate=2,
                    skip_type='sum', bn_kwargs=bn_kwargs)
                self.__setattr__('middleflow_block{}'.format(i), block)

            self.exitflow_block1 = XceptionBlock(
                728, [728, 1024, 1024], stride=1, dilate=2,
                skip_type='conv', bn_kwargs=bn_kwargs)
            self.exitflow_block2 = XceptionBlock(
                1024, [1536, 1536, 2048], stride=1, dilate=4,
                skip_type='none', bn_kwargs=bn_kwargs, activ_first=False,
                dw_activ_list=[F.relu, F.relu, F.relu],
                pw_activ_list=[F.relu, F.relu, F.relu])

    def __call__(self, x):
        h = self.entryflow_conv1(x)
        h = self.entryflow_conv2(h)
        h, _ = self.entryflow_block1(h)
        h, lowlevel = self.entryflow_block2(h)
        h, _ = self.entryflow_block3(h)

        for i in range(1, 17):
            h, _ = self['middleflow_block{}'.format(i)](h)

        h, _ = self.exitflow_block1(h)
        highlevel, _ = self.exitflow_block2(h)

        return lowlevel, highlevel
