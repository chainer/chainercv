from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links.connection import Conv2DBNActiv
from chainercv.links.connection import SeparableConv2DBNActiv
from chainercv.links.model.deeplab.aspp import SeparableASPP
from chainercv.links.model.deeplab.xception import Xception65

import numpy as np


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
        crop = (h, w)

        # Pad image with mean pixel value.
        mean_pixel = np.array(
            self.feature_extractor.mean_pixel, dtype=np.float32)
        bg = np.zeros((3, h, w), dtype=np.float32) + mean_pixel[:, None, None]
        bg[:, :H, :W] = image
        image = bg

        # scale to [-1.0, 1.0]
        image = image / 127.5 - 1.0

        return image, crop

    def __call__(self, x):
        lowlevel, highlevel = self.feature_extractor(x)
        highlevel = self.aspp(highlevel)
        h = self.decoder(lowlevel, highlevel)
        return h

    def predict(self, imgs):
        labels = []
        for img in imgs:
            C, H, W = img.shape
            img, crop = self._prepare(img)

            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                x = self.__call__(x)
                score = F.resize_images(x, crop)[0, :, :H, :W].array
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
