from __future__ import division

import itertools
import numpy as np
import six

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv import transforms


class SSD(chainer.Chain):
    variance = (0.1, 0.2)

    nms_threshold = 0.45
    score_threshold = 0.6

    conv_init = {
        'initialW': initializers.GlorotUniform(),
        'initial_bias': initializers.Zero(),
    }

    def __init__(self, n_class, **links):
        self.n_class = n_class

        super(SSD, self).__init__(
            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )
        for name, link in six.iteritems(links):
            self.add_link(name, link)

        for ar in self.aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(
                None, n * 4, 3, pad=1, **self.conv_init))
            self.conf.add_link(L.Convolution2D(
                None, n * (self.n_class + 1), 3, pad=1, **self.conv_init))

        # the format of default_bbox is (center_x, center_y, width, height)
        self._default_bbox = list()
        for k in range(len(self.grids)):
            for v, u in itertools.product(range(self.grids[k]), repeat=2):
                cx = (u + 0.5) * self.steps[k]
                cy = (v + 0.5) * self.steps[k]

                s = self.sizes[k]
                self._default_bbox.append((cx, cy, s, s))

                s = np.sqrt(self.sizes[k] * self.sizes[k + 1])
                self._default_bbox.append((cx, cy, s, s))

                s = self.sizes[k]
                for ar in self.aspect_ratios[k]:
                    self._default_bbox.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    self._default_bbox.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        self._default_bbox = np.stack(self._default_bbox)

    def to_cpu(self):
        super(SSD, self).to_cpu()
        self._default_bbox = chainer.cuda.to_cpu(self._default_bbox)

    def to_gpu(self):
        super(SSD, self).to_gpu()
        self._default_bbox = chainer.cuda.to_gpu(self._default_bbox)

    def features(self, x):
        raise NotImplementedError

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
                conf, (conf.shape[0], -1, self.n_class + 1))
            ys_conf.append(conf)

        y_loc = F.concat(ys_loc, axis=1)
        y_conf = F.concat(ys_conf, axis=1)

        return y_loc, y_conf

    def __call__(self, x):
        return self._multibox(self.features(x))

    def prepare(self, img):
        return NotImplementedError

    def _decode(self, loc, conf):
        xp = self.xp
        # the format of bbox is (center_x, center_y, width, height)
        bboxes = xp.dstack((
            self._default_bbox[:, :2] +
            loc[:, :, :2] * self.variance[0] * self._default_bbox[:, 2:],
            self._default_bbox[:, 2:] *
            xp.exp(loc[:, :, 2:] * self.variance[1])))
        # convert the format of bbox to (x_min, y_min, x_max, y_max)
        bboxes[:, :, :2] -= bboxes[:, :, 2:] / 2
        bboxes[:, :, 2:] += bboxes[:, :, :2]
        scores = xp.exp(conf)
        scores /= scores.sum(axis=2, keepdims=True)
        return bboxes, scores

    def _suppress(self, raw_bbox, raw_score):
        xp = self.xp

        raw_bbox = chainer.cuda.to_cpu(raw_bbox)
        raw_score = chainer.cuda.to_cpu(raw_score)

        bbox = list()
        label = list()
        score = list()
        for i in range(1, 1 + self.n_class):
            mask = raw_score[:, i] >= self.score_threshold
            bbox_label = raw_bbox[mask]
            score_label = raw_score[mask, i]

            if self.nms_threshold is not None:
                order = score_label.argsort()[::-1]
                bbox_label, score_label = bbox_label[order], score_label[order]
                bbox_label, param = transforms.non_maximum_suppression(
                    bbox_label, self.nms_threshold, return_param=True)
                score_label = score_label[param['selection']]

            bbox.append(bbox_label)
            label.append((i,) * len(bbox_label))
            score.append(score_label)

        bbox = xp.array(np.vstack(bbox).astype(np.float32))
        label = xp.array(np.hstack(label).astype(np.int32))
        score = xp.array(np.hstack(score).astype(np.float32))

        return bbox, label, score

    def predict(self, imgs):
        prepared_imgs = list()
        sizes = list()
        for img in imgs:
            _, H, W = img.shape
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            sizes.append((W, H))

        prepared_imgs = self.xp.stack(prepared_imgs)
        loc, conf = self(prepared_imgs)
        raw_bboxes, raw_scores = self._decode(loc.data, conf.data)

        bboxes = list()
        labels = list()
        scores = list()
        for raw_bbox, raw_score, size in zip(raw_bboxes, raw_scores, sizes):
            raw_bbox = transforms.resize_bbox(raw_bbox, (1, 1), size)
            bbox, label, score = self._suppress(raw_bbox, raw_score)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores
