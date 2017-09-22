import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links import PickableSequentialChain


class BuildingBlock(PickableSequentialChain):

    """A building block that consists of several Bottleneck layers.

    input --> BottleneckA --> BottleneckB * (n_layer - 1) --> output

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
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW,
                stride_first)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)


class BottleneckA(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): The number of channels of input arrays.
        mid_channels (int): The number of channels of intermediate arrays.
        out_channels (int): The number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        stride_first (bool): If :obj:`True`, apply strided convolution
            with the first convolution layer. Otherwise, apply
            strided convolution with the second convolution layer.

    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None, stride_first=False):
        if stride_first:
            first_stride = stride
            second_stride = 1
        else:
            first_stride = 1
            second_stride = stride

        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, first_stride, 0,
                initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, second_stride, 1,
                initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(chainer.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): The number of channels of input and output arrays.
        mid_channels (int): The number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.

    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)
