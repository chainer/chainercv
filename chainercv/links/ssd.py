from __future__ import division

import itertools
import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L
from chainer import serializers

from chainercv import transforms


class _Normalize(chainer.Link):

    def __init__(self, n_channels, initial=0, eps=1e-5):
        super(_Normalize, self).__init__()
        self.eps = eps
        self.add_param(
            'scale', n_channels,
            initializer=initializers._get_initializer(initial))

    def __call__(self, x):
        x = F.normalize(x, eps=self.eps, axis=1)
        scale = F.broadcast_to(self.scale[:, np.newaxis, np.newaxis], x.shape)
        return x * scale


class _SSDVGG16(chainer.Chain):
    mean = (104, 117, 123)
    variance = (0.1, 0.2)

    conv_init = {
        'initialW': initializers.GlorotUniform(),
        'initial_bias': initializers.Zero(),
    }
    norm_init = {
        'initial': initializers.Constant(20),
    }

    def __init__(self, n_classes):
        self.n_classes = n_classes

        super(_SSDVGG16, self).__init__(
            conv1_1=L.Convolution2D(None, 64, 3, pad=1, **self.conv_init),
            conv1_2=L.Convolution2D(None, 64, 3, pad=1, **self.conv_init),

            conv2_1=L.Convolution2D(None, 128, 3, pad=1, **self.conv_init),
            conv2_2=L.Convolution2D(None, 128, 3, pad=1, **self.conv_init),

            conv3_1=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),
            conv3_2=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),
            conv3_3=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),

            conv4_1=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            conv4_2=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            conv4_3=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            norm4=_Normalize(512, **self.norm_init),

            conv5_1=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),
            conv5_2=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),
            conv5_3=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),

            conv6=L.DilatedConvolution2D(
                None, 1024, 3, pad=6, dilate=6, **self.conv_init),
            conv7=L.Convolution2D(None, 1024, 1, **self.conv_init),

            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )
        for ar in self.aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(
                None, n * 4, 3, pad=1, **self.conv_init))
            self.conf.add_link(L.Convolution2D(
                None, n * (self.n_classes + 1), 3, pad=1, **self.conv_init))

        # the format of default_bbox is (center_x, center_y, width, height)
        self.default_bbox = self._default_bbox()

    def _features(self, x):
        ys = list()

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        ys.append(self.norm4(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        ys.append(h)

        return ys

    def _multibox(self, xs):
        ys_loc = list()
        ys_conf = list()
        for i, x in enumerate(xs):
            loc = self.loc[i](x)
            loc = F.transpose(loc, (0, 2, 3, 1))
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            ys_loc.append(loc)

            conf = self.conf[i](x)
            conf = F.transpose(conf, (0, 2, 3, 1))
            conf = F.reshape(
                conf, (conf.shape[0], -1, self.n_classes + 1))
            ys_conf.append(conf)

        y_loc = F.concat(ys_loc, axis=1)
        y_conf = F.concat(ys_conf, axis=1)

        return y_loc, y_conf

    def __call__(self, x):
        return self._multibox(self._features(x))

    def _default_bbox(self):
        bbox = list()
        for k in range(len(self.grids)):
            for v, u in itertools.product(range(self.grids[k]), repeat=2):
                cx = (u + 0.5) * self.steps[k]
                cy = (v + 0.5) * self.steps[k]

                s = self.sizes[k]
                bbox.append((cx, cy, s, s))

                s = np.sqrt(self.sizes[k] * self.sizes[k + 1])
                bbox.append((cx, cy, s, s))

                s = self.sizes[k]
                for ar in self.aspect_ratios[k]:
                    bbox.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    bbox.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        return np.stack(bbox)

    def _decode(self, loc, conf):
        # the format of bbox is (center_x, center_y, width, height)
        bbox = np.hstack((
            self.default_bbox[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_bbox[:, 2:],
            self.default_bbox[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        # convert the format of bbox to (x_min, y_min, x_max, y_max)
        bbox[:, :2] -= bbox[:, 2:] / 2
        bbox[:, 2:] += bbox[:, :2]
        conf = np.exp(conf)
        conf /= conf.sum(axis=1, keepdims=True)
        return bbox, conf

    def _prepare(self, img):
        H, W = img.shape[1:]
        img = transforms.resize(img, (self.insize, self.insize))
        img -= np.array(self.mean)[:, np.newaxis, np.newaxis]
        return img, (W, H)

    def _suppress(self, bbox, conf, conf_threshold, nms_threshold):
        bbox_all = list()
        label_all = list()
        conf_all = list()
        for label in range(1, 1 + self.n_classes):
            mask = conf[:, label] >= conf_threshold
            bbox_label, conf_label = bbox[mask], conf[mask, label]

            if nms_threshold is not None:
                order = conf_label.argsort()[::-1]
                bbox_label, conf_label = bbox_label[order], conf_label[order]
                bbox_label, param = transforms.non_maximum_suppression(
                    bbox_label, nms_threshold, return_param=True)
                conf_label = conf_label[param['selection']]

            bbox_all.append(bbox_label)
            label_all.append((label,) * len(bbox_label))
            conf_all.append(conf_label)

        return np.vstack(bbox_all), np.hstack(label_all), np.hstack(conf_all)

    def predict(self, img, conf_threshold=0.01, nms_threshold=0.45):
        img, size = self._prepare(img)
        loc, conf = self(img[np.newaxis])
        bbox, conf = self._decode(loc.data[0], conf.data[0])
        bbox = transforms.resize_bbox(bbox, (1, 1), size)
        return self._suppress(bbox, conf, conf_threshold, nms_threshold)


class SSD300(_SSDVGG16):

    insize = 300
    grids = (38, 19, 10, 5, 3, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)]

    def __init__(self, n_classes, pretrained_model):
        super(SSD300, self).__init__(n_classes)

        self.add_link(
            'conv8_1', L.Convolution2D(None, 256, 1, **self.conv_init))
        self.add_link(
            'conv8_2',
            L.Convolution2D(None, 512, 3, stride=2, pad=1, **self.conv_init))

        self.add_link(
            'conv9_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv9_2',
            L.Convolution2D(None, 256, 3, stride=2, pad=1, **self.conv_init))

        self.add_link(
            'conv10_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv10_2', L.Convolution2D(None, 256, 3, **self.conv_init))

        self.add_link(
            'conv11_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv11_2', L.Convolution2D(None, 256, 3, **self.conv_init))

        if pretrained_model:
            serializers.load_npz(pretrained_model, self)

    def _features(self, x):
        ys = super(SSD300, self)._features(x)
        for i in range(8, 11 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys


class SSD512(_SSDVGG16):

    insize = 512
    grids = (64, 32, 16, 8, 4, 2, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2, ))
    steps = [s / 512 for s in (8, 16, 32, 64, 128, 256, 512)]
    sizes = [s / 512 for s in
             (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)]

    def __init__(self, n_classes, pretrained_model=None):
        super(SSD512, self).__init__(n_classes)

        self.add_link(
            'conv8_1', L.Convolution2D(None, 256, 1, **self.conv_init))
        self.add_link(
            'conv8_2',
            L.Convolution2D(None, 512, 3, stride=2, pad=1, **self.conv_init))

        for i in range(9, 11 + 1):
            self.add_link(
                'conv{:d}_1'.format(i),
                L.Convolution2D(None, 128, 1, **self.conv_init))
            self.add_link(
                'conv{:d}_2'.format(i),
                L.Convolution2D(
                    None, 256, 3, stride=2, pad=1, **self.conv_init))

        self.add_link(
            'conv12_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv12_2',
            L.Convolution2D(None, 256, 4,  pad=1, **self.conv_init))

        if pretrained_model:
            serializers.load_npz(pretrained_model, self)

    def _features(self, x):
        ys = super(SSD512, self)._features(x)
        for i in range(8, 12 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys
