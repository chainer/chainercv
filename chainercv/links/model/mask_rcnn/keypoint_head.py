from __future__ import division

import numpy as np
import PIL

import cv2

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda
from chainer.initializers import HeNormal
from chainer.initializers import Normal

from chainercv.links import Conv2DActiv
from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox


class KeypointHead(chainer.Chain):

    _canonical_scale = 224
    _roi_size = 14
    _roi_sample_ratio = 2
    map_size = 56

    def __init__(self, n_point, scales):
        super(KeypointHead, self).__init__()

        initialW = HeNormal(1, fan_option='fan_out')
        with self.init_scope():
            self.conv1 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.conv2 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.conv3 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.conv4 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.conv5 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.conv6 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.conv7 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.conv8 = Conv2DActiv(512, 3, pad=1, initialW=initialW)
            self.point = L.Deconvolution2D(
                n_point, 4, pad=1, stride=2, initialW=initialW)

        self._scales = scales
        self.n_point = n_point

    def __call__(self, hs, rois, roi_indices):
        pooled_hs = []
        for l, h in enumerate(hs):
            if len(rois[l]) == 0:
                continue

            pooled_hs.append(F.roi_average_align_2d(
                h, rois[l], roi_indices[l],
                self._roi_size,
                self._scales[l], self._roi_sample_ratio))

        if len(pooled_hs) == 0:
            return chainer.Variable(
               self.xp.empty(
                   (0, self.n_point, self.map_size, self.map_size),
                   dtype=np.float32))

        h = F.concat(pooled_hs, axis=0)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.conv8(h)
        h = self.point(h)
        return F.resize_images(h, (self.map_size, self.map_size))

    def distribute(self, rois, roi_indices):
        # Compleetely same as MaskHead.distribute
        size = self.xp.sqrt(self.xp.prod(rois[:, 2:] + 1 - rois[:, :2], axis=1))
        level = self.xp.floor(self.xp.log2(
            size / self._canonical_scale + 1e-6)).astype(np.int32)
        # skip last level
        level = self.xp.clip(
            level + len(self._scales) // 2, 0, len(self._scales) - 2)

        masks = [level == l for l in range(len(self._scales))]
        rois = [rois[mask] for mask in masks]
        roi_indices = [roi_indices[mask] for mask in masks]
        order = self.xp.argsort(
            self.xp.concatenate([self.xp.where(mask)[0] for mask in masks]))
        return rois, roi_indices, order

    def decode(self, point_maps, bboxes):
        points = []
        point_scores = []
        for bbox, point_map in zip(bboxes, point_maps):
            point = np.zeros((len(bbox), self.n_point, 2), dtype=np.float32)
            point_score = np.zeros((len(bbox), self.n_point), dtype=np.float32)

            hs = bbox[:, 2] - bbox[:, 0]
            ws = bbox[:, 3] - bbox[:, 1]
            h_ceils = np.ceil(np.maximum(hs, 1))
            w_ceils = np.ceil(np.maximum(ws, 1))
            h_corrections = hs / h_ceils
            w_corrections = ws / w_ceils
            for i, (bb, point_m) in enumerate(zip(bbox, point_map)):
                point_m = cv2.resize(
                    point_m.transpose((1, 2, 0)),
                    (w_ceils[i], h_ceils[i]),
                    interpolation=cv2.INTER_CUBIC).transpose(
                        (2, 0, 1))
                _, H, W = point_m.shape
                for k in range(self.n_point):
                    pos = point_m[k].argmax()
                    x_int = pos % W
                    y_int = (pos - x_int) // W

                    y = (y_int + 0.5) * h_corrections[i]
                    x = (x_int + 0.5) * w_corrections[i]
                    point[i, k, 0] = y + bb[0]
                    point[i, k, 1] = x + bb[1]
                    point_score[i, k] = point_m[k, y_int, x_int]
            points.append(point)
            point_scores.append(point_score)
        return points, point_scores
