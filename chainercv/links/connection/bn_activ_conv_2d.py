import chainer
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D


class BNActivConv2D(chainer.Chain):
    """Batch Normalization --> Activation --> Convolution2D

    This is a chain that sequentially aplies a batch normalization,
    an activation and a two-dimensional convolution.

    The arguments are the same as that of
    :class:`chainer.links.Convolution2D`
    except for :obj:`activ` and :obj:`bn_kwargs`.
    Note that the default value for the :obj:`nobias`
    is changed to :obj:`True`.

    Unlike :class:`chainer.links.Convolution2D`, this class requires
    :obj:`in_channels` defined explicitly.

    >>> l = BNActivConv2D(5, 10, 3)

    Args:
        in_channels (int): The number of channels of input arrays.
            This needs to be explicitly defined.
        out_channels (int): The number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            :obj:`ksize=k` and :obj:`ksize=(k, k)` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            :obj:`stride=s` and :obj:`stride=(s, s)` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            :obj:`pad=p` and :obj:`pad=(p, p)` are equivalent.
        nobias (bool): If :obj:`True`,
            then this link does not use the bias term.
        initialW (4-D array): Initial weight value. If :obj:`None`, the default
            initializer is used.
            May also be a callable that takes :obj:`numpy.ndarray` or
            :obj:`cupy.ndarray` and edits its value.
        initial_bias (1-D array): Initial bias value. If :obj:`None`, the bias
            is set to 0.
            May also be a callable that takes :obj:`numpy.ndarray` or
            :obj:`cupy.ndarray` and edits its value.
        activ (callable): An activation function. The default value is
            :func:`chainer.functions.relu`.
        bn_kwargs (dict): Keyword arguments passed to initialize
            :class:`chainer.links.BatchNormalization`.

    """

    def __init__(self, in_channels, out_channels, ksize=None,
                 stride=1, pad=0, nobias=True, initialW=None,
                 initial_bias=None, activ=relu, bn_kwargs=dict()):
        self.activ = activ
        super(BNActivConv2D, self).__init__()
        with self.init_scope():
            self.bn = BatchNormalization(in_channels, **bn_kwargs)
            self.conv = Convolution2D(
                in_channels, out_channels, ksize, stride, pad,
                nobias, initialW, initial_bias)

    def __call__(self, x):
        h = self.bn(x)
        h = self.activ(h)
        return self.conv(h)
