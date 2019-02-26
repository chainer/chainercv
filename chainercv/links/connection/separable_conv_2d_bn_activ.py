import chainer
from chainer.functions import identity
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D

try:
    from chainermn.links import MultiNodeBatchNormalization
except ImportError:
    pass


class SeparableConv2DBNActiv(chainer.Chain):

    """Separable Convolution with batch normalization and activation.

    Convolution2D(Depthwise) --> Batch Normalization --> Activation
    --> Convolution2D(Pointwise) --> Batch Normalization --> Activation

    Separable convolution with batch normalizations and activations.
    Parameters are almost same as :class:`Conv2DBNActiv` except
    depthwise and pointwise convolution parameters.

    Args:
        in_channels (int): Number of channels of input arrays.
            Unlike :class:`Conv2DBNActiv`, this can't accept
            :obj:`None` currently.
        out_channels (int): Number of channels of output arrays.
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            :obj:`ksize=k` and :obj:`ksize=(k, k)` are equivalent.
        stride (int or tuple of ints): Stride of filter applications.
            :obj:`stride=s` and :obj:`stride=(s, s)` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            :obj:`pad=p` and :obj:`pad=(p, p)` are equivalent.
        dilate (int or tuple of ints): Dilation factor of filter applications.
            :obj:`dilate=d` and :obj:`dilate=(d, d)` are equivalent.
        nobias (bool): If :obj:`True`,
            then this link does not use the bias term.
        dw_initialW (callable): Initial weight value of depthwise convolution.
            If :obj:`None`, the default initializer is used.
            May also be a callable that takes :obj:`numpy.ndarray` or
            :obj:`cupy.ndarray` and edits its value.
        pw_initialW (callable): Initial weight value of pointwise convolution.
        dw_initial_bias (callable): Initial bias value of depthwise
            convolution. If :obj:`None`, the bias is set to 0.
            May also be a callable that takes :obj:`numpy.ndarray` or
            :obj:`cupy.ndarray` and edits its value.
        pw_initial_bias (callable): Initial bias value of pointwise
            convolution.
        dw_activ (callable): An activation function of depthwise convolution.
            The default value is :func:`chainer.functions.relu`. If this is
            :obj:`None`, no activation is applied (i.e. the activation is the
            identity function).
        pw_activ (callable): An activation function of pointwise convolution.
        bn_kwargs (dict): Keyword arguments passed to initialize
            :class:`chainer.links.BatchNormalization`. If a ChainerMN
            communicator (:class:`~chainermn.communicators.CommunicatorBase`)
            is given with the key :obj:`comm`,
            :obj:`~chainermn.links.MultiNodeBatchNormalization` will be used
            for the batch normalization. Otherwise,
            :obj:`~chainer.links.BatchNormalization` will be used.

    """

    def __init__(self, in_channels, out_channels, ksize,
                 stride=1, pad=0, dilate=1, nobias=False,
                 dw_initialW=None, pw_initialW=None,
                 dw_initial_bias=None, pw_initial_bias=None,
                 dw_activ=identity, pw_activ=relu, bn_kwargs={}):

        self.dw_activ = identity if dw_activ is None else dw_activ
        self.pw_activ = identity if pw_activ is None else pw_activ
        super(SeparableConv2DBNActiv, self).__init__()

        with self.init_scope():
            self.depthwise = Convolution2D(
                in_channels, in_channels, ksize=ksize, stride=stride,
                pad=pad, dilate=dilate, groups=in_channels,
                nobias=nobias, initialW=dw_initialW)
            self.pointwise = Convolution2D(
                in_channels, out_channels, 1,
                nobias=nobias, initialW=pw_initialW)

            if 'comm' in bn_kwargs:
                self.dw_bn = MultiNodeBatchNormalization(
                    in_channels, **bn_kwargs)
                self.pw_bn = MultiNodeBatchNormalization(
                    out_channels, **bn_kwargs)
            else:
                self.dw_bn = BatchNormalization(in_channels, **bn_kwargs)
                self.pw_bn = BatchNormalization(out_channels, **bn_kwargs)

    def __call__(self, x):
        h = self.depthwise(x)
        h = self.dw_bn(h)
        h = self.dw_activ(h)

        h = self.pointwise(h)
        h = self.pw_bn(h)
        h = self.pw_activ(h)

        return h
