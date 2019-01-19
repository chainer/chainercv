from __future__ import division

import chainer
import chainer.functions as F

from chainercv.links import Conv2DBNActiv
from chainercv.links import SeparableConv2DBNActiv


class SeparableASPP(chainer.Chain):
    def __init__(self, in_channels, out_channels=256,
                 dilate_list=[12, 24, 36], bn_kwargs={}):
        super(SeparableASPP, self).__init__()

        with self.init_scope():
            self.image_pooling_conv = Conv2DBNActiv(
                in_channels, out_channels, 1, bn_kwargs=bn_kwargs)
            self.conv1x1 = Conv2DBNActiv(
                in_channels, out_channels, 1, bn_kwargs=bn_kwargs)
            self.atrous1 = SeparableConv2DBNActiv(
                in_channels, out_channels, 3, 1,
                dilate_list[0], dilate_list[0], nobias=True,
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.atrous2 = SeparableConv2DBNActiv(
                in_channels, out_channels, 3, 1,
                dilate_list[1], dilate_list[1], nobias=True,
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.atrous3 = SeparableConv2DBNActiv(
                in_channels, out_channels, 3, 1,
                dilate_list[2], dilate_list[2], nobias=True,
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.proj = Conv2DBNActiv(
                out_channels * 5, out_channels, 1, bn_kwargs=bn_kwargs)

    def image_pooling(self, x):
        _, _, H, W = x.shape
        x = F.average(x, axis=(2, 3), keepdims=True)
        x = self.image_pooling_conv(x)
        B, C, _, _ = x.shape
        x = F.broadcast_to(x, (B, C, H, W))
        return x

    def __call__(self, x):
        h = []
        h.append(self.image_pooling(x))
        h.append(self.conv1x1(x))
        h.append(self.atrous1(x))
        h.append(self.atrous2(x))
        h.append(self.atrous3(x))
        h = F.concat(h, axis=1)
        h = self.proj(h)
        h = F.dropout(h)

        return h
