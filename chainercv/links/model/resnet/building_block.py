import chainer
import chainer.functions as F

from chainercv.links import Conv2DBNActiv

from chainercv.links import PickableSequentialChain


class BuildingBlock(PickableSequentialChain):

    """A building block that consists of several Bottleneck layers.

    input --> Bottleneck (shortcut) --> Bottleneck * (n_layer - 1) --> output

    Args:
        n_layer (int): The number of layers used in the building block.
        in_channels (int): The number of channels of input arrays.
        mid_channels (int): The number of channels of intermediate arrays.
        out_channels (int): The number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        stride_first (bool): If :obj:`True`, apply strided convolution
            with the first convolution layer of the BottleneckA layer.
            Otherwise, apply strided convolution with the
            second convolution layer.

    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None, stride_first=False):
        shortcut = Conv2DBNActiv(in_channels, out_channels, 1, stride, 0,
                                 nobias=True, initialW=initialW,
                                 activ=lambda x: x)
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = Bottleneck(
                in_channels, mid_channels, out_channels, stride, initialW,
                shortcut=shortcut, stride_first=stride_first)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = Bottleneck(out_channels, mid_channels,
                                        out_channels, initialW=initialW)
                setattr(self, name, bottleneck)


class Bottleneck(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): The number of channels of input arrays.
        mid_channels (int): The number of channels of intermediate arrays.
        out_channels (int): The number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        shortcut (callable): The residual is calculated with this.
            If this is :obj:`None`, the residual is the same
            as the input.
        stride_first (bool): If :obj:`True`, apply strided convolution
            with the first convolution layer. Otherwise, apply
            strided convolution with the second convolution layer.

    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, initialW=None, shortcut=None,
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
                                       bn_kwargs={'initial_gamma': 0},
                                       activ=lambda x: x)
            if shortcut is not None:
                self.shortcut = shortcut

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        if hasattr(self, 'shortcut'):
            residual = self.shortcut(x)
        else:
            residual = x
        h += residual
        h = F.relu(h)
        return h
