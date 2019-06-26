import chainer
from chainer.functions.connection import convolution_2d
from chainer.functions import mean
from chainer.functions import relu
from chainer.functions import rsqrt
from chainer.links import BatchNormalization
from chainer.links import Convolution2D

try:
    from chainermn.links import MultiNodeBatchNormalization
except ImportError:
    pass


def rstd(x, axis, keepdims, eps):
    return rsqrt(mean(x ** 2, axis=axis, keepdims=keepdims) + eps)


class Convolution2DWS(Convolution2D):

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])

        W_mean = mean(self.W, axis=(1, 2, 3), keepdims=True)
        W_minus_mean = self.W - W_mean
        W_rstd = rstd(W_minus_mean, axis=(1, 2, 3), keepdims=True, eps=1e-5)
        W_norm = W_rstd * W_minus_mean

        return convolution_2d.convolution_2d(
            x, W_norm, self.b, self.stride, self.pad, dilate=self.dilate,
            groups=self.groups)


class Conv2DBNActiv(chainer.Chain):
    """Convolution2D --> Batch Normalization --> Activation

    This is a chain that sequentially applies a two-dimensional convolution,
    a batch normalization and an activation.

    The arguments are the same as that of
    :class:`chainer.links.Convolution2D`
    except for :obj:`activ` and :obj:`bn_kwargs`.
    :obj:`bn_kwargs` can include :obj:`comm` key and a communicator of
    ChainerMN as the value to use
    :class:`chainermn.links.MultiNodeBatchNormalization`. If
    :obj:`comm` is not included in :obj:`bn_kwargs`,
    :class:`chainer.links.BatchNormalization` link from Chainer is used.
    Note that the default value for the :obj:`nobias`
    is changed to :obj:`True`.

    Example:

        There are several ways to initialize a :class:`Conv2DBNActiv`.

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
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            :obj:`ksize=k` and :obj:`ksize=(k, k)` are equivalent.
        stride (int or tuple of ints): Stride of filter applications.
            :obj:`stride=s` and :obj:`stride=(s, s)` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            :obj:`pad=p` and :obj:`pad=(p, p)` are equivalent.
        dilate (int or tuple of ints): Dilation factor of filter applications.
            :obj:`dilate=d` and :obj:`dilate=(d, d)` are equivalent.
        groups (int): The number of groups to use grouped convolution. The
            default is one, where grouped convolution is not used.
        nobias (bool): If :obj:`True`,
            then this link does not use the bias term.
        initialW (callable): Initial weight value. If :obj:`None`, the default
            initializer is used.
            May also be a callable that takes :obj:`numpy.ndarray` or
            :obj:`cupy.ndarray` and edits its value.
        initial_bias (callable): Initial bias value. If :obj:`None`, the bias
            is set to 0.
            May also be a callable that takes :obj:`numpy.ndarray` or
            :obj:`cupy.ndarray` and edits its value.
        activ (callable): An activation function. The default value is
            :func:`chainer.functions.relu`. If this is :obj:`None`,
            no activation is applied (i.e. the activation is the identity
            function).
        bn_kwargs (dict): Keyword arguments passed to initialize
            :class:`chainer.links.BatchNormalization`. If a ChainerMN
            communicator (:class:`~chainermn.communicators.CommunicatorBase`)
            is given with the key :obj:`comm`,
            :obj:`~chainermn.links.MultiNodeBatchNormalization` will be used
            for the batch normalization. Otherwise,
            :obj:`~chainer.links.BatchNormalization` will be used.

    """

    def __init__(self, in_channels, out_channels, ksize=None,
                 stride=1, pad=0, dilate=1, groups=1, nobias=True,
                 initialW=None, initial_bias=None,
                 weight_standarization=False, activ=relu, bn_kwargs={}):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.activ = activ
        super(Conv2DBNActiv, self).__init__()
        with self.init_scope():
            if weight_standarization:
                self.conv = Convolution2DWS(
                    in_channels, out_channels, ksize, stride, pad,
                    nobias, initialW, initial_bias, dilate=dilate,
                    groups=groups)
            else:
                self.conv = Convolution2D(
                    in_channels, out_channels, ksize, stride, pad,
                    nobias, initialW, initial_bias, dilate=dilate,
                    groups=groups)
            if 'comm' in bn_kwargs:
                self.bn = MultiNodeBatchNormalization(
                    out_channels, **bn_kwargs)
            else:
                self.bn = BatchNormalization(out_channels, **bn_kwargs)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activ is None:
            return h
        else:
            return self.activ(h)
