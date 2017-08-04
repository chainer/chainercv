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

    Example:

        There are sevaral ways to make a Convolution2DBlock chain.

        1. Give the first three arguments explicitly:

            >>> l = Convolution2DBlock(5, 10, 3)

        2. Omit :obj:`in_channels` or fill it with :obj:`None`:

            In these ways, attributes are initialized at runtime based on
            the channel size of the input.

            >>> l = Convolution2DBlock(None, 10, 3)
            >>> l = Convolution2DBlock(10, 3)


    Args:
        in_channels (int or None): Number of channels of input arrays.
            If :obj:`None`, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            :obj:`ksize=k` and :obj:`ksize=(k, k)` are equivalent.
        activation (callable): An activation function. The default value is
            :func:`chainer.functions.relu`.
        use_bn (bool): Indicates whether to use batch normalization or not.
            The default value is :obj:`False`.
        conv_kwargs (dict): Key-word arguments passed to initialize
            :class:`chainer.links.Convolution2D`.
        bn_kwargs (dict): Key-word arguments passed to initialize
            :class:`chainer.links.BatchNormalization`.

    """

    def __init__(self, in_channels, out_channels, ksize=None,
                 activation=relu, use_bn=False,
                 conv_kwargs=dict(), bn_kwargs=dict()):
        self.use_bn = use_bn
        self.activation = activation
        super(Convolution2DBlock, self).__init__()
        with self.init_scope():
            self.conv = Convolution2D(in_channels, out_channels, ksize,
                                      **conv_kwargs)
            if self.use_bn:
                self.bn = BatchNormalization(out_channels, **bn_kwargs)

    def __call__(self, x):
        h = self.conv(x)
        if self.use_bn:
            h = self.bn(h)
        return self.activation(h)
