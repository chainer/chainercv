import chainer
import chainer.functions as F

from chainercv.links import Conv2DBNActiv


def _leaky_relu(x):
    return F.leaky_relu(x, slope=0.1)


class Block(chainer.ChainList):
    def __init__(self, in_channels, n_block):
        super().__init__(Conv2DBNActiv(
            in_channels * 2, 3, stride=2, pad=1, activ=_leaky_relu))
        for _ in range(n_block):
            self.add_link(Conv2DBNActiv(in_channels, 1, activ=_leaky_relu))
            self.add_link(
                Conv2DBNActiv(in_channels * 2, 3, pad=1, activ=_leaky_relu))

    def __call__(self, x):
        it = iter(self)
        h = next(it)(x)
        for a, b in zip(it, it):
            h += b(a(h))
        return h


class Darknet53(chainer.Sequential):
    def __init__(self):
        super().__init__(Conv2DBNActiv(32, 3, pad=1, activ=_leaky_relu))
        for k, n_block in enumerate((1, 2, 8, 8, 4)):
            self.append(Block(32 << k, n_block))
