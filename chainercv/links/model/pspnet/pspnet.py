from math import ceil

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import warnings

try:
    from chainermn.links import MultiNodeBatchNormalization
except:
    warnings.warn('To perform batch normalization with multiple GPUs or '
                  'multiple nodes, MultiNodeBatchNormalization link is '
                  'needed. Please install ChainerMN: '
                  'pip install pip install git+git://github.com/chainer/'
                  'chainermn.git@distributed-batch-normalization')


class ConvBNReLU(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1):
        super().__init__()
        comm = chainer.config.comm
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            if dilation > 1:
                self.conv = L.DilatedConvolution2D(
                    in_ch, out_ch, ksize, stride, pad, dilation, True, w)
            else:
                self.conv = L.Convolution2D(
                    in_ch, out_ch, ksize, stride, pad, True, w)
            if comm is not None:
                self.bn = MultiNodeBatchNormalization(out_ch, comm)
            else:
                self.bn = L.BatchNormalization(out_ch)

    def __call__(self, x, relu=True):
        h = self.bn(self.conv(x))
        return h if not relu else F.relu(h)


class PyramidPoolingModule(chainer.ChainList):

    def __init__(self, in_ch, feat_size, pyramids):
        super().__init__(
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0))
        self.ksizes = (feat_size // np.array(pyramids)).tolist()

    def __call__(self, x):
        ys = [x]
        h, w = x.shape[2:]
        for f, ksize in zip(self, self.ksizes):
            ksize = (ksize, ksize)

            # Calc padding sizes
            h_rem, w_rem = h % ksize[0], w % ksize[1]
            pad_h = int(ceil((ksize[0] - h_rem) / 2.0)) if h_rem > 0 else 0
            pad_w = int(ceil((ksize[1] - w_rem) / 2.0)) if w_rem > 0 else 0
            pad = (pad_h, pad_w)

            y = F.average_pooling_2d(x, ksize, ksize, pad)
            y = f(y)  # Reduce num of channels
            y = F.resize_images(y, (h, w))
            ys.append(y)
        return F.concat(ys, axis=1)


class BottleneckConv(chainer.Chain):

    def __init__(self, in_ch, mid_ch, out_ch, stride=2, dilate=False):
        mid_stride = chainer.config.mid_stride
        super().__init__()
        with self.init_scope():
            self.cbr1 = ConvBNReLU(
                in_ch, mid_ch, 1, 1 if mid_stride else stride, 0)
            if dilate:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, dilate, dilate)
            else:
                self.cbr2 = ConvBNReLU(
                    mid_ch, mid_ch, 3, stride if mid_stride else 1, 1)
            self.cbr3 = ConvBNReLU(mid_ch, out_ch, 1, 1, 0)
            self.cbr4 = ConvBNReLU(in_ch, out_ch, 1, stride, 0)

    def __call__(self, x):
        h = self.cbr1(x)
        h = self.cbr2(h)
        h1 = self.cbr3(h, relu=False)
        h2 = self.cbr4(x, relu=False)
        return F.relu(h1 + h2)


class BottleneckIdentity(chainer.Chain):

    def __init__(self, in_ch, mid_ch, dilate=False):
        super().__init__()
        with self.init_scope():
            self.cbr1 = ConvBNReLU(in_ch, mid_ch, 1, 1, 0)
            if dilate:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, dilate, dilate)
            else:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
            self.cbr3 = ConvBNReLU(mid_ch, in_ch, 1, 1, 0)

    def __call__(self, x):
        h = self.cbr1(x)
        h = self.cbr2(h)
        h = self.cbr3(h, relu=False)
        return F.relu(h + x)


class ResBlock(chainer.ChainList):

    def __init__(self, n_layer, in_ch, mid_ch, out_ch, stride):
        super().__init__()
        self.add_link(BottleneckConv(in_ch, mid_ch, out_ch, stride))
        for _ in range(1, n_layer):
            self.add_link(BottleneckIdentity(out_ch, mid_ch))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class DilatedResBlock(chainer.ChainList):

    def __init__(self, n_layer, in_ch, mid_ch, out_ch, dilate):
        super().__init__()
        self.add_link(BottleneckConv(in_ch, mid_ch, out_ch, 1, dilate))
        for _ in range(1, n_layer):
            self.add_link(BottleneckIdentity(out_ch, mid_ch, dilate))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class DilatedFCN(chainer.Chain):

    def __init__(self, n_blocks):
        super().__init__()
        with self.init_scope():
            self.cbr1_1 = ConvBNReLU(None, 64, 3, 2, 1)
            self.cbr1_2 = ConvBNReLU(64, 64, 3, 1, 1)
            self.cbr1_3 = ConvBNReLU(64, 128, 3, 1, 1)
            self.res2 = ResBlock(n_blocks[0], 128, 64, 256, 1)
            self.res3 = ResBlock(n_blocks[1], 256, 128, 512, 2)
            self.res4 = DilatedResBlock(n_blocks[2], 512, 256, 1024, 2)
            self.res5 = DilatedResBlock(n_blocks[3], 1024, 512, 2048, 4)

    def __call__(self, x):
        h = self.cbr1_3(self.cbr1_2(self.cbr1_1(x)))  # 1/2
        h = F.max_pooling_2d(h, 3, 2, 1)  # 1/4
        h = self.res2(h)
        h = self.res3(h)  # 1/8
        if chainer.config.train:
            h1 = self.res4(h)
            h2 = self.res5(h1)
            return h1, h2
        else:
            h = self.res4(h)
            return self.res5(h)


class PSPNet(chainer.Chain):

    """Pyramid Scene Parsing Network

    This Chain supports any depth of ResNet and any pyramid levels for
    the pyramid pooling module (PPM).

    Args:
        n_class (int): The number of channels in the last convolution layer.
        n_blocks (list of int): Numbers of layers in ResNet. Typically,
            [3, 4, 23, 3] for ResNet101 (used for PASCAL VOC2012 and
            Cityscapes in the original paper) and [3, 4, 6, 3] for ResNet50
            (used for ADE20K datset in the original paper).
        feat_size (int): The feature map size of the output of the base ResNet.
            Typically it will be 1/8 of the input image size.
        pyramids (list of int): The number of division to the feature map in
            each pyramid level. The length of this list will be the number of
            levels of pyramid in the pyramid pooling module.
        mid_stride (bool): If True, spatial dimention reduction in bottleneck
            modules in ResNet part will be done at the middle 3x3 convolution.
            It means that the stride of the middle 3x3 convolution will be two.
            Otherwise (if it's set to False), the stride of the first 1x1
            convolution in the bottleneck module will be two as in the original
            ResNet and Deeplab v2.
        comm (ChainerMN communicator or None): If a ChainerMN communicator is
            given, it will be used for distributed batch normalization during
            training. If None, all batch normalization links will not share
            the input vectors among GPUs before calculating mean and variance.
            The original PSPNet implementation uses distributed batch
            normalization.

    """

    def __init__(self, n_class, n_blocks=[3, 4, 23, 3], feat_size=90,
                 pyramids=[1, 2, 3, 6], mid_stride=True, comm=None):
        chainer.config.mid_stride = mid_stride
        chainer.config.comm = comm
        super().__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.trunk = DilatedFCN(n_blocks=n_blocks)

            # To calculate auxirally loss
            if chainer.config.train:
                self.cbr_aux = ConvBNReLU(None, 512, 3, 1, 1)
                self.out_aux = L.Convolution2D(
                    512, n_class, 3, 1, 1, False, w)

            # Main branch
            self.ppm = PyramidPoolingModule(2048, feat_size, pyramids)
            self.cbr_main = ConvBNReLU(4096, 512, 3, 1, 1)
            self.out_main = L.Convolution2D(512, n_class, 1, 1, 0, False, w)

    def __call__(self, x):
        if chainer.config.train:
            aux, h = self.trunk(x)
            aux = F.dropout(self.cbr_aux(aux), ratio=0.1)
            aux = self.out_aux(aux)
            aux = F.resize_images(aux, x.shape[2:])
        else:
            h = self.trunk(x)

        h = self.ppm(h)
        h = F.dropout(self.cbr_main(h), ratio=0.1)
        h = self.out_main(h)
        h = F.resize_images(h, x.shape[2:])

        if chainer.config.train:
            return aux, h
        else:
            return h

    def predict(self, imgs):
        with chainer.using_config('train', False):
            imgs = chainer.Variable(self.xp.asarray(imgs))
            y = self.__call__(imgs)
            label = F.argmax(y, axis=1).data
            label = chainer.cuda.to_cpu(label)
            return label
