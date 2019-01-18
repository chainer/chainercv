from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links import Conv2DBNActiv
from chainercv.links import SeparableConv2DBNActiv


class XceptionBlock(chainer.Chain):
    def __init__(self, in_channels, depthlist, stride=1, dilate=1,
                 skip_type='conv', activ_first=True, bn_kwargs={},
                 dw_activ_list=[F.identity, F.identity, F.identity],
                 pw_activ_list=[F.relu, F.relu, F.identity]):
        super(XceptionBlock, self).__init__()
        self.skip_type = skip_type
        self.activ_first = activ_first
        self.separable2_activ = pw_activ_list[1]

        with self.init_scope():
            self.separable1 = SeparableConv2DBNActiv(
                in_channels, depthlist[0], 3, 1,
                dilate, dilate, bn_kwargs=bn_kwargs,
                dw_activ=dw_activ_list[0], pw_activ=pw_activ_list[0])
            self.separable2 = SeparableConv2DBNActiv(
                depthlist[0], depthlist[1], 3, 1,
                dilate, dilate, bn_kwargs=bn_kwargs,
                dw_activ=dw_activ_list[1], pw_activ=F.identity)
            self.separable3 = SeparableConv2DBNActiv(
                depthlist[1], depthlist[2], 3, stride,
                dilate, dilate, bn_kwargs=bn_kwargs,
                dw_activ=dw_activ_list[2], pw_activ=pw_activ_list[2])
            if skip_type == 'conv':
                self.conv = Conv2DBNActiv(
                    in_channels, depthlist[2], 1, activ=F.identity,
                    nobias=True, stride=stride, bn_kwargs=bn_kwargs)

    def __call__(self, x):
        if self.activ_first:
            h = F.relu(x)
        else:
            h = x

        h = self.separable1(h)
        h = self.separable2(h)
        separable2 = h
        h = self.separable2_activ(h)
        h = self.separable3(h)

        if self.skip_type == 'conv':
            skip = self.conv(x)
            h = h + skip
        elif self.skip_type == 'sum':
            h = h + x
        elif self.skip_type == 'none':
            pass

        if not self.activ_first:
            h = F.relu(h)

        return h, separable2


class Xception65(chainer.Chain):
    mean_pixel = [127.5, 127.5, 127.5]

    def __init__(self, bn_kwargs={}):
        super(Xception65, self).__init__()

        with self.init_scope():
            self.entryflow_conv1 = Conv2DBNActiv(
                3, 32, 3, 2, 1, bn_kwargs=bn_kwargs)
            self.entryflow_conv2 = Conv2DBNActiv(
                32, 64, 3, 1, 1, bn_kwargs=bn_kwargs)
            self.entryflow_block1 = XceptionBlock(
                64, [128, 128, 128], stride=2,
                skip_type='conv', bn_kwargs=bn_kwargs)
            self.entryflow_block2 = XceptionBlock(
                128, [256, 256, 256], stride=2,
                skip_type='conv', bn_kwargs=bn_kwargs)
            self.entryflow_block3 = XceptionBlock(
                256, [728, 728, 728], stride=1,
                skip_type='conv', bn_kwargs=bn_kwargs)

            for i in range(1, 17):
                block = XceptionBlock(
                    728, [728, 728, 728], stride=1, dilate=2,
                    skip_type='sum', bn_kwargs=bn_kwargs)
                self.__setattr__('middleflow_block{}'.format(i), block)

            self.exitflow_block1 = XceptionBlock(
                728, [728, 1024, 1024], stride=1, dilate=2,
                skip_type='conv', bn_kwargs=bn_kwargs)
            self.exitflow_block2 = XceptionBlock(
                1024, [1536, 1536, 2048], stride=1, dilate=4,
                skip_type='none', bn_kwargs=bn_kwargs, activ_first=False,
                dw_activ_list=[F.relu, F.relu, F.relu],
                pw_activ_list=[F.relu, F.relu, F.relu])

    def __call__(self, x):
        h = self.entryflow_conv1(x)
        h = self.entryflow_conv2(h)
        h, _ = self.entryflow_block1(h)
        h, lowlevel = self.entryflow_block2(h)
        h, _ = self.entryflow_block3(h)

        for i in range(1, 17):
            h, _ = self['middleflow_block{}'.format(i)](h)

        h, _ = self.exitflow_block1(h)
        highlevel, _ = self.exitflow_block2(h)

        return lowlevel, highlevel


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
                dilate_list[0], dilate_list[0],
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.atrous2 = SeparableConv2DBNActiv(
                in_channels, out_channels, 3, 1,
                dilate_list[1], dilate_list[1],
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.atrous3 = SeparableConv2DBNActiv(
                in_channels, out_channels, 3, 1,
                dilate_list[2], dilate_list[2],
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


class Decoder(chainer.Chain):
    def __init__(self, in_channels, out_channels, proj_channels=48,
                 depth_channels=256, dilate_list=[12, 24, 36], bn_kwargs={}):
        super(Decoder, self).__init__()

        with self.init_scope():
            self.feature_proj = Conv2DBNActiv(in_channels, proj_channels, 1)
            self.conv1 = SeparableConv2DBNActiv(
                in_channels+proj_channels, depth_channels, 3, 1, 1, 1,
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.conv2 = SeparableConv2DBNActiv(
                depth_channels, depth_channels, 3, 1, 1, 1,
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.conv_logits = L.Convolution2D(
                depth_channels, out_channels, 1, 1, 0)

    def __call__(self, x, pool):
        x = self.feature_proj(x)
        pool = F.resize_images(pool, x.shape[2:])
        h = F.concat((pool, x), axis=1)
        h = self.conv1(h)
        h = self.conv2(h)
        logits = self.conv_logits(h)

        return logits


class DeepLabV3plus(chainer.Chain):
    def __init__(self, feature_extractor, aspp, decoder, crop=(513, 513)):
        super(DeepLabV3plus, self).__init__()
        self.crop = crop

        with self.init_scope():
            self.feature_extractor = feature_extractor
            self.aspp = aspp
            self.decoder = decoder

        self.feature_extractor.pick = 'entryflow_block2', 'exitflow_block2'

    def _prepare(self, image):
        _, H, W = image.shape

        # Pad image and label to have dimensions >= [crop_height, crop_width]
        h = max(self.crop[0], H)
        w = max(self.crop[1], W)

        # Pad image with mean pixel value.
        mean_pixel = np.array(
            self.feature_extractor.mean_pixel, dtype=np.float32)
        bg = np.zeros((3, h, w), dtype=np.float32) + mean_pixel[:, None, None]
        bg[:, :H, :W] = image
        image = bg

        # scale to [-1.0, 1.0]
        image = image / 127.5 - 1.0

        return image

    def __call__(self, x):
        lowlevel, highlevel = self.feature_extractor(x)
        highlevel = self.aspp(highlevel)
        h = self.decoder(lowlevel, highlevel)
        return h

    def predict(self, imgs):
        labels = []
        for img in imgs:
            C, H, W = img.shape
            img = self._prepare(img)

            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                x = self.__call__(x)
                score = F.resize_images(x, self.crop)[0, :, :H, :W].array
            score = chainer.backends.cuda.to_cpu(score)
            label = np.argmax(score, axis=0).astype(np.int32)
            labels.append(label)
        return labels


class DeepLabV3plusXception65(DeepLabV3plus):
    def __init__(self, n_class):
        super(DeepLabV3plusXception65, self).__init__(
            Xception65(bn_kwargs={'decay': 0.9997, 'eps': 1e-3}),
            SeparableASPP(2048, 256, bn_kwargs={'decay': 0.9997, 'eps': 1e-5}),
            Decoder(256, n_class, bn_kwargs={'decay': 0.9997, 'eps': 1e-5}))
