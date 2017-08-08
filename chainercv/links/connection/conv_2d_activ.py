import chainer
from chainer.functions import relu
from chainer.links import Convolution2D


class Conv2DActiv(chainer.Chain):
    """Convolution2D --> Activation

    This is a chain that does two-dimensional convolution
    and applies an activation.

    Example:

        There are sevaral ways to make a Conv2DActiv chain.

        1. Give the first three arguments explicitly:

            >>> l = Conv2DActiv(5, 10, 3)

        2. Omit :obj:`in_channels` or fill it with :obj:`None`:

            In these ways, attributes are initialized at runtime based on
            the channel size of the input.

            >>> l = Conv2DActiv(None, 10, 3)
            >>> l = Conv2DActiv(10, 3)

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

    """

    def __init__(self, in_channels, out_channels, ksize=None,
                 stride=1, pad=0, nobias=False, initialW=None,
                 initial_bias=None, activ=relu):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.activ = activ
        super(Conv2DActiv, self).__init__()
        with self.init_scope():
            self.conv = Convolution2D(
                in_channels, out_channels, ksize, stride, pad,
                nobias, initialW, initial_bias)

    def __call__(self, x):
        h = self.conv(x)
        return self.activ(h)
