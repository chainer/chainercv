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
        for l in self:
            h = l(h)
        h += x
        return h


class YOLOv3(chainer.ChainList):

    insize = 416

    def __init__(self, n_fg_class):
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
            for _ in range(3):
                self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))
                self.append(Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu))
            self.append(Convolution2D((1 + 4 + n_fg_class) * 3, 1))

    def __call__(self, x):
        ys = []

        h = x
        hs = []
        for i, l in enumerate(self):
            h = self[i](h)
            if i in {35, 43, 51}:
                ys.append(h)

            # shortcut and route
            if i in {14, 23, 33, 41}:
                hs.append(h)
            # shortcut
            elif i in {35, 43}:
                h = hs.pop()
            # route
            elif i in {36, 44}:
                h = F.concat((_upsample(h), hs.pop()))

        return ys
