import chainer
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D


class SeparableConv2DBNActiv(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize,
                 stride=1, pad=0, dilate=1, nobias=False,
                 depthwise_activ=None, pointwise_activ=relu, bn_kwargs={}):

        self.depthwise_activ = depthwise_activ
        self.pointwise_activ = pointwise_activ
        super(SeparableConv2DBNActiv, self).__init__()

        with self.init_scope():
            self.depthwise = Convolution2D(
                in_channels, in_channels, ksize=ksize, stride=stride,
                pad=pad, dilate=dilate, groups=in_channels, nobias=True)
            self.depthwise_bn = BatchNormalization(in_channels, **bn_kwargs)
            self.pointwise = Convolution2D(
                in_channels, out_channels, 1, nobias=True)
            self.pointwise_bn = BatchNormalization(out_channels, **bn_kwargs)

    def __call__(self, x):
        h = self.depthwise(x)
        h = self.depthwise_bn(h)
        if self.depthwise_activ is not None:
            h = self.depthwise_activ(h)

        h = self.pointwise(h)
        h = self.pointwise_bn(h)
        if self.pointwise_activ is not None:
            h = self.pointwise_activ(h)

        return h
