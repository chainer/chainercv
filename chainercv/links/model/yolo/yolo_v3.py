import chainer
import chainer.functions as F
from chainer.links import Convolution2D

from chainercv.links import Conv2DBNActiv


def _leaky_relu(x):
    return F.leaky_relu(x, slope=0.1)


def _upsample(x):
    return F.unpooling_2d(x, 2, cover_all=False)


class ResidualBlock(chainer.ChainList):

    def __init__(self, *links):
        super().__init__(*links)

    def __call__(self, x):
        h = x
        for link in self:
            h = link(h)
        h += x
        return h


class Darknet53Extractor(chainer.ChainList):

    insize = 416

    def __init__(self):
        super().__init__()

        # Darknet53
        self.append(Conv2DBNActiv(32, 3, pad=1, activ=_leaky_relu))
        for k, n_block in enumerate((1, 2, 8, 8, 4)):
            self.append(Conv2DBNActiv(
                32 << (k + 1), 3, stride=2, pad=1, activ=_leaky_relu))
            for _ in range(n_block):
                self.append(ResidualBlock(
                    Conv2DBNActiv(32 << k, 1, activ=_leaky_relu),
                    Conv2DBNActiv(32 << (k + 1), 3, pad=1, activ=_leaky_relu)))

        # additional links
        for i, n in enumerate((512, 256, 128)):
            if i > 0:
                self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))

    def __call__(self, x):
        ys = []
        h = x
        hs = []
        for i, link in enumerate(self):
            h = link(h)
            if i in {33, 39, 45}:
                ys.append(h)
            elif i in {14, 23}:
                hs.append(h)
            elif i in {34, 40}:
                h = F.concat((_upsample(h), hs.pop()))
        return ys


class YOLOv3(chainer.Chain):

    def __init__(self, n_fg_class):
        super().__init__()
        self.n_fg_class = n_fg_class

        with self.init_scope():
            self.extractor = Darknet53Extractor()
            self.subnet = chainer.ChainList()

        for n in (512, 256, 128):
            self.subnet.append(chainer.Sequential(
                Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu),
                Convolution2D(3 * (1 + 4 + self.n_fg_class), 1)))

    @property
    def insize(self):
        return self.extractor.insize

    def __call__(self, x):
        ys = []
        for i, h in enumerate(self.extractor(x)):
            h = self.subnet[i](h)
            h = F.reshape(h, (h.shape[0], 3, 1 + 4 + self.n_fg_class, -1))
            h = F.transpose(h, (0, 3, 1, 2))
            ys.append(h)
        return F.concat(ys)
