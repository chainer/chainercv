import chainer
from chainer.functions import identity
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D


class SeparableConv2DBNActiv(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize,
                 stride=1, pad=0, dilate=1, nobias=False,
                 dw_activ=identity, pw_activ=relu, bn_kwargs={}):

        self.dw_activ = dw_activ
        self.pw_activ = pw_activ
        super(SeparableConv2DBNActiv, self).__init__()

        with self.init_scope():
            self.depthwise = Convolution2D(
                in_channels, in_channels, ksize=ksize, stride=stride,
                pad=pad, dilate=dilate, groups=in_channels, nobias=True)
            self.dw_bn = BatchNormalization(in_channels, **bn_kwargs)
            self.pointwise = Convolution2D(
                in_channels, out_channels, 1, nobias=True)
            self.pw_bn = BatchNormalization(out_channels, **bn_kwargs)

    def __call__(self, x):
        h = self.depthwise(x)
        h = self.dw_bn(h)
        h = self.dw_activ(h)

        h = self.pointwise(h)
        h = self.pw_bn(h)
        h = self.pw_activ(h)

        return h
