import chainer
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D


class Conv2DBNActiv(chainer.Chain):
    """Convolution2D --> Batch Normalization --> Activation

    This is a chain that sequentially apllies a two-dimensional convolution,
    a batch normalization and an activation.

    The arguments are the same as that of
    :class:`chainer.links.Convolution2D`
    except for :obj:`activ` and :obj:`bn_kwargs`.
    Note that the default value for the :obj:`nobias`
    is changed to :obj:`True`.

    Example:

        There are sevaral ways to initialize a :class:`Conv2DBNActiv`.

        1. Give the first three arguments explicitly:

            >>> l = Conv2DBNActiv(5, 10, 3)

        2. Omit :obj:`in_channels` or fill it with :obj:`None`:

            In these ways, attributes are initialized at runtime based on
            the channel size of the input.

            >>> l = Conv2DBNActiv(10, 3)
            >>> l = Conv2DBNActiv(None, 10, 3)

    Args:
        in_channels (int or None): Number of channels of input arrays.
            If :obj:`None`, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
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
                 initial_bias=None, activ=relu, bn_kwargs={}):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.activ = activ
        super(Conv2DBNActiv, self).__init__()
        with self.init_scope():
            self.conv = Convolution2D(
                in_channels, out_channels, ksize, stride, pad,
                nobias, initialW, initial_bias)
            self.bn = BatchNormalization(out_channels, **bn_kwargs)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return self.activ(h)
