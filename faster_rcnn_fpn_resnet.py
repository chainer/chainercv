import numpy as np

import chainer.functions as F

import chainercv

from fpn import Head
from fpn import FasterRCNN
from fpn.fpn import FPN
from fpn.rpn import RPN


def _make_fpn(cls, mean):
    base = cls(n_class=1, arch='he')
    base.pick = ('res2', 'res3', 'res4', 'res5')
    base.remove_unused()

    assert(mean in {'chainercv', 'detectron'})
    if mean == 'detectron':
        base.mean = np.array((122.7717, 115.9465, 102.9801))[:, None, None]
    base.pool1 = lambda x: F.max_pooling_2d(
        x, 3, stride=2, pad=1, cover_all=False)

    return FPN(base, len(base.pick), (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64))


class FasterRCNNFPNResNet50(FasterRCNN):

    def __init__(self, n_fg_class, mean):
        extractor = _make_fpn(chainercv.links.ResNet50, mean)
        super().__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_fg_class + 1, extractor.scales),
        )


class FasterRCNNFPNResNet101(FasterRCNN):

    def __init__(self, n_fg_class, mean):
        extractor = _make_fpn(chainercv.links.ResNet101, mean)
        super().__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_fg_class + 1, extractor.scales),
        )
