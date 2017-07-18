import numpy as np

import chainer
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D


class Convolution2DBlock(chainer.Chain):
    """Convolution2D --> (Batch Normalization) --> Activation

    This is a chain that does two-dimensional convolution
    and applies an activation.
    Optionally, batch normalization can be executed in the middle.

    The parameters are a combination of the ones for
    :class:`chainer.links.Convolution2D` and
    :class:`chainer.links.BatchNormalization` except for
    :obj:`activation` and :obj:`use_bn`.

    :obj:`activation` is a callable. The default value is
    :func:`chainer.functions.relu`.

    :obj:`use_bn` is a bool that indicates whether to use
    batch normalization or not. The default value is :obj:`False`.

    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 activation=relu, use_bn=False,
                 nobias=False, initialW=None, initial_bias=None,
                 decay=0.9, eps=2e-5, dtype=np.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        self.use_bn = use_bn
        self.activation = activation
        super(Convolution2DBlock, self).__init__()
        with self.init_scope():
            self.conv = Convolution2D(in_channels, out_channels, ksize, stride,
                                      pad, nobias, initialW, initial_bias)
            if self.use_bn:
                self.bn = BatchNormalization(
                    out_channels, decay, eps, dtype, use_gamma, use_beta,
                    initial_gamma, initial_beta)

    def __call__(self, x):
        h = self.conv(x)
        if self.use_bn:
            h = self.bn(h)
        return self.activation(h)
