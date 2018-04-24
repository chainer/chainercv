import chainer
import chainer.functions as F

from chainercv.links import Conv2DBNActiv


def _leaky_relu(x):
    return F.leaky_relu(x, slope=0.1)


class ResidualBlock(chainer.ChainList):

    def __init__(self, *links):
        super().__init__(*links)

    def __call__(self, x):
        h = x
        for l in self:
            h = l(h)
        h += x
        return h


class Darknet53(chainer.Sequential):

    def __init__(self):
        super().__init__(Conv2DBNActiv(32, 3, pad=1, activ=_leaky_relu))
        for k, n_block in enumerate((1, 2, 8, 8, 4)):
            self.append(Conv2DBNActiv(
                32 << (k + 1), 3, stride=2, pad=1, activ=_leaky_relu))
            for _ in range(n_block):
                self.append(ResidualBlock(
                    Conv2DBNActiv(32 << k, 1, activ=_leaky_relu),
                    Conv2DBNActiv(32 << (k + 1), 3, pad=1, activ=_leaky_relu)))
