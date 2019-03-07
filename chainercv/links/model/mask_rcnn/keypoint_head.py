from __future__ import division

import numpy as np
import PIL

import cv2

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda
from chainer.initializers import HeNormal

from chainercv.links import Conv2DActiv
from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou

from chainercv.links.model.mask_rcnn.misc import point_to_roi_points


# make a bilinear interpolation kernel
# credit @longjon
def _upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)


class KeypointHead(chainer.Chain):

    _canonical_scale = 224
    _roi_size = 14
    _roi_sample_ratio = 2
    point_map_size = 56

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
            # Do not update the weight of this link
            self.upsample = L.Deconvolution2D(
                n_point, n_point, 4, pad=1, stride=2, nobias=True)
        self.upsample.W.data[:] = 0
        self.upsample.W.data[np.arange(n_point), np.arange(n_point)] = _upsample_filt(4)

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
                   (0, self.n_point, self.point_map_size, self.point_map_size),
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
        return self.upsample(h)

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


def keypoint_loss_pre(rois, roi_indices, gt_points, gt_visibles,
                      gt_bboxes, gt_head_labels, point_map_size):
    _, n_point, _ = gt_points[0].shape

    xp = cuda.get_array_module(*rois)

    n_level = len(rois)

    roi_levels = xp.hstack(
        xp.array((l,) * len(rois[l])) for l in range(n_level)).astype(np.int32)
    rois = xp.vstack(rois).astype(np.float32)
    roi_indices = xp.hstack(roi_indices).astype(np.int32)
    gt_head_labels = xp.hstack(gt_head_labels)

    index = (gt_head_labels > 0).nonzero()[0]
    point_roi_levels = roi_levels[index]
    point_rois = rois[index]
    point_roi_indices = roi_indices[index]

    gt_roi_points = xp.empty(
        (len(point_rois), n_point, 2), dtype=np.float32)
    gt_roi_visibles = xp.empty(
        (len(point_rois), n_point), dtype=np.bool)
    for i in np.unique(cuda.to_cpu(point_roi_indices)):
        gt_point = gt_points[i]
        gt_visible = gt_visibles[i]
        gt_bbox = gt_bboxes[i]

        index = (point_roi_indices == i).nonzero()[0]
        point_roi = point_rois[index]
        iou = bbox_iou(point_roi, gt_bbox)
        gt_index = iou.argmax(axis=1)
        gt_roi_point, gt_roi_visible = point_to_roi_points(
                gt_point[gt_index], gt_visible[gt_index],
                point_roi, point_map_size)
        gt_roi_points[index] = xp.array(gt_roi_point)
        gt_roi_visibles[index] = xp.array(gt_roi_visible)

    flag_masks = [point_roi_levels == l for l in range(n_level)]
    point_rois = [point_rois[m] for m in flag_masks]
    point_roi_indices = [point_roi_indices[m] for m in flag_masks]
    gt_roi_points = [gt_roi_points[m] for m in flag_masks]
    gt_roi_visibles = [gt_roi_visibles[m] for m in flag_masks]
    return point_rois, point_roi_indices, gt_roi_points, gt_roi_visibles


def keypoint_loss_post(
        point_maps, point_roi_indices, gt_roi_points,
        gt_roi_visibles, batchsize):
    xp = cuda.get_array_module(point_maps.array)

    point_roi_indices = xp.hstack(point_roi_indices).astype(np.int32)
    gt_roi_points = xp.vstack(gt_roi_points).astype(np.int32)
    gt_roi_visibles = xp.vstack(gt_roi_visibles).astype(np.bool)

    B, K, H, W = point_maps.shape
    point_maps = point_maps.reshape((B * K, H * W))
    spatial_labels = gt_roi_points[:, :, 0] * W + gt_roi_points[:, :, 1]
    spatial_labels = spatial_labels.reshape((B * K,))
    spatial_labels[xp.logical_not(gt_roi_visibles.reshape((B * K,)))] = -1
    # Remember that the loss is normalized by the total number of
    # visible keypoints.
    keypoint_loss = F.softmax_cross_entropy(point_maps, spatial_labels)
    return keypoint_loss
