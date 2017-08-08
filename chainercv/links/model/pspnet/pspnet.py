from math import ceil

import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links import DistributedBatchNormalization

_comm = None


class ConvBNReLU(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1):
        global _comm
        super().__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            if dilation > 1:
                self.conv = L.DilatedConvolution2D(
                    in_ch, out_ch, ksize, stride, pad, dilation, True, w)
            else:
                self.conv = L.Convolution2D(
                    in_ch, out_ch, ksize, stride, pad, True, w)
            if _comm is not None:
                self.bn = DistributedBatchNormalization(out_ch, _comm)
            else:
                self.bn = L.BatchNormalization(out_ch)

    def __call__(self, x, relu=True):
        h = self.bn(self.conv(x))
        return h if not relu else F.relu(h)


class PyramidPoolingModule(chainer.ChainList):

    def __init__(self, in_ch):
        self.denominators = [1, 2, 3, 6]
        super().__init__(
            ConvBNReLU(in_ch, in_ch // len(self.denominators), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(self.denominators), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(self.denominators), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(self.denominators), 1, 1, 0)
        )

    def __call__(self, x):
        ys = [x]
        h, w = x.shape[2:]
        for f, deno in zip(self, self.denominators):
            ksize = (h // deno, w // deno)

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

    def __init__(self, in_ch, mid_ch, out_ch, stride=2, dilation=False):
        super().__init__()
        with self.init_scope():
            self.cbr1 = ConvBNReLU(in_ch, mid_ch, 1, stride, 0)
            if dilation:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 2, 2)
            else:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
            self.cbr3 = ConvBNReLU(mid_ch, out_ch, 1, 1, 0)
            self.cbr4 = ConvBNReLU(in_ch, out_ch, 1, stride, 0)

    def __call__(self, x):
        h = self.cbr1(x)
        h = self.cbr2(h)
        h1 = self.cbr3(h, relu=False)
        h2 = self.cbr4(x, relu=False)
        return F.relu(h1 + h2)


class BottleneckIdentity(chainer.Chain):

    def __init__(self, in_ch, mid_ch, dilation=False):
        super().__init__()
        with self.init_scope():
            self.cbr1 = ConvBNReLU(in_ch, mid_ch, 1, 1, 0)
            if dilation:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 2, 2)
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

    def __init__(self, n_layer, in_ch, mid_ch, out_ch):
        super().__init__()
        self.add_link(BottleneckConv(in_ch, mid_ch, out_ch, 1, dilation=True))
        for _ in range(1, n_layer):
            self.add_link(BottleneckIdentity(out_ch, mid_ch, dilation=True))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class DilatedFCN(chainer.Chain):

    def __init__(self, n_blocks=[3, 4, 6, 3]):
        super().__init__()
        with self.init_scope():
            self.cbr1_1 = ConvBNReLU(None, 64, 3, 2, 1)
            self.cbr1_2 = ConvBNReLU(64, 64, 3, 1, 1)
            self.cbr1_3 = ConvBNReLU(64, 128, 3, 1, 1)
            self.res2 = ResBlock(n_blocks[0], 128, 64, 256, 1)
            self.res3 = ResBlock(n_blocks[1], 256, 128, 512, 2)
            self.res4 = DilatedResBlock(n_blocks[2], 512, 256, 1024)
            self.res5 = DilatedResBlock(n_blocks[3], 1024, 512, 2048)

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

    101: [3, 4, 23, 3]
    50: [3, 4, 6, 3]

    # of parameters: 51609576

    """

    def __init__(self, n_class, n_blocks=[3, 4, 6, 3], comm=None):
        global _comm
        _comm = comm
        super().__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.resnet = DilatedFCN(n_blocks)

            # To calculate auxiliary loss
            if chainer.config.train:
                self.cbr_aux = ConvBNReLU(None, 512, 3, 1, 1)
                self.out_aux = L.Convolution2D(
                    512, n_class, 3, 1, 1, False, w)

            # Main branch
            self.ppm = PyramidPoolingModule(2048)
            self.cbr_main = ConvBNReLU(None, 512, 3, 1, 1)
            self.out_main = L.Convolution2D(None, n_class, 3, 1, 1, False, w)

    def __call__(self, x):
        if chainer.config.train:
            aux, h = self.resnet(x)
            aux = F.dropout(self.cbr_aux(aux), ratio=0.1)
            aux = self.out_aux(aux)
            aux = F.resize_images(aux, x.shape[2:])
        else:
            h = self.resnet(x)

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
