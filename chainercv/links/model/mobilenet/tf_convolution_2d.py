import numpy as np

import chainer
from chainer.functions import pad
from chainer.links import Convolution2D
from chainer.utils import conv


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


def _get_pad(in_size, ksize, stride, tf_padding):
    if tf_padding == 'SAME':
        tf_out_size = int(np.ceil(float(in_size) / stride))
    elif tf_padding == 'VALID':
        tf_out_size = int(np.ceil(float(in_size - ksize + 1) / stride))
    pad = int(stride * tf_out_size - in_size + ksize - stride)
    assert conv.get_conv_outsize(in_size + pad, ksize, stride,
                                 0) == tf_out_size
    return pad


def _tf_padding(x, ksize, stride, tf_padding):
    pad_2 = _get_pad(x.shape[2], ksize[0], stride[0], tf_padding)
    pad_3 = _get_pad(x.shape[3], ksize[1], stride[1], tf_padding)
    if pad_2 or pad_3:
        return pad(
            x,
            ((0, 0), (0, 0), (pad_2 // 2, int(np.ceil(float(pad_2) / 2))),
             (pad_3 // 2, int(np.ceil(float(pad_3) / 2)))),
            mode='constant')
    else:
        return x


class TFConvolution2D(chainer.Chain):
    """Tensorflow compatible Convolution2D

    This is a Convolution2D chain that imitates Tensorflow's tf.nn.conv2d.

    The arguments are the same as that of
    :class:`chainer.links.Convolution2D` except for `pad`.
    :obj:`pad` can be set TF's "SAME" or "VALID" in addition to integer value.
    If integer value is set,
    this chain is equal to :class:`chainer.links.Convolution2D`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=None,
                 stride=1,
                 pad='SAME',
                 nobias=False,
                 initialW=None,
                 initial_bias=None,
                 **kwargs):
        super(TFConvolution2D, self).__init__()
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        if pad in ('SAME', 'VALID'):  # TF compatible pad
            self.padding = lambda x: _tf_padding(x, _pair(self.conv.ksize),
                                                 _pair(self.conv.stride), pad)
            conv_pad = 0
        else:
            self.padding = None
            assert isinstance(pad, int)
            conv_pad = pad

        with self.init_scope():
            self.conv = Convolution2D(in_channels, out_channels, ksize, stride,
                                      conv_pad, nobias, initialW, initial_bias,
                                      **kwargs)

    @property
    def W(self):
        return self.conv.W

    @property
    def b(self):
        return self.conv.b

    def forward(self, x):
        if self.padding is None:
            return self.conv(x)
        else:
            return self.conv(self.padding(x))
