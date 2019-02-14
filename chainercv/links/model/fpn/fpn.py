import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class FPN(chainer.Chain):
    """An extractor class of Feature Pyramid Networks.

    This class wraps a feature extractor and provides
    multi-scale features.

    Args:
        base (Link): A base feature extractor.
            It should have :meth:`__call__` and :obj:`mean`.
            :meth:`__call__` should take a batch of images and return
            feature maps of them. The size of the :math:`k+1`-th feature map
            should be the half as that of the :math:`k`-th feature map.
        n_base_output (int): The number of feature maps
            that :obj:`base` returns.
        scales (tuple of floats): The scales of feature maps.

    """

    def __init__(self, base, n_base_output, scales):
        super(FPN, self).__init__()
        with self.init_scope():
            self.base = base
            self.inner = chainer.ChainList()
            self.outer = chainer.ChainList()

        init = {'initialW': initializers.GlorotNormal()}
        for _ in range(n_base_output):
            self.inner.append(L.Convolution2D(256, 1, **init))
            self.outer.append(L.Convolution2D(256, 3, pad=1, **init))

        self.scales = scales

    @property
    def mean(self):
        return self.base.mean

    def __call__(self, x):
        hs = list(self.base(x))

        for i in reversed(range(len(hs))):
            hs[i] = self.inner[i](hs[i])
            if i + 1 < len(hs):
                hs[i] += F.unpooling_2d(hs[i + 1], 2, cover_all=False)

        for i in range(len(hs)):
            hs[i] = self.outer[i](hs[i])

        while len(hs) < len(self.scales):
            hs.append(F.max_pooling_2d(hs[-1], 1, stride=2, cover_all=False))

        return hs
