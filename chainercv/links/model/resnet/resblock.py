import chainer
import chainer.functions as F

from chainercv.links import Conv2DBNActiv


class ResBlock(chainer.Chain):

    """A building block for ResNets.

    in --> Bottleneck with shortcut --> Bottleneck * (n_layer - 1) --> out

    Args:
        n_layer (int): The number of layers used in the building block.
        in_channels (int): The number of channels of input arrays.
        mid_channels (int): The number of channels of intermediate arrays.
        out_channels (int): The number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        stride_first (bool): This determines the behavior of the
            bottleneck with a shortcut. If :obj:`True`, apply strided
            convolution with the first convolution layer.
            Otherwise, apply strided convolution with the
            second convolution layer.

    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None, stride_first=False):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.a = Bottleneck(
                in_channels, mid_channels, out_channels, stride,
                initialW, residual_conv=True, stride_first=stride_first)
            self._forward = ['a']
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = Bottleneck(
                    out_channels, mid_channels, out_channels, stride=1,
                    initialW=initialW, residual_conv=False)
                self.add_link(name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x


class Bottleneck(chainer.Chain):

    """A bottleneck layer.

    Args:
        in_channels (int): The number of channels of input arrays.
        mid_channels (int): The number of channels of intermediate arrays.
        out_channels (int): The number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        residual_conv (bool): If :obj:`True`, apply a 1x1 convolution
            to the residual.
        stride_first (bool): If :obj:`True`, apply strided convolution
            with the first convolution layer. Otherwise, apply
            strided convolution with the second convolution layer.

    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, initialW=None, residual_conv=False,
                 stride_first=False):
        if stride_first:
            first_stride = stride
            second_stride = 1
        else:
            first_stride = 1
            second_stride = stride
        super(Bottleneck, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(in_channels, mid_channels, 1,
                                       first_stride, 0, initialW=initialW,
                                       nobias=True)
            self.conv2 = Conv2DBNActiv(mid_channels, mid_channels, 3,
                                       second_stride, 1, initialW=initialW,
                                       nobias=True)
            self.conv3 = Conv2DBNActiv(mid_channels, out_channels, 1, 1, 0,
                                       initialW=initialW, nobias=True,
                                       activ=lambda x: x)
            if residual_conv:
                self.residual_conv = Conv2DBNActiv(
                    in_channels, out_channels, 1, stride, 0,
                    nobias=True, initialW=initialW, activ=lambda x: x)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(x)
        else:
            residual = x
        h += residual
        h = F.relu(h)
        return h
