from __future__ import division

import numpy as np
import PIL

import cv2

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer.initializers import HeNormal
import chainer.links as L

from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox


class MaskHead(chainer.Chain):

    _canonical_scale = 224
    _roi_size = 14
    _roi_sample_ratio = 2
    mask_size = _roi_size * 2

    # Remember, initialization is MSRAFill
    def __init__(self, n_class, scales):
        super(MaskHead, self).__init__()

        initialW = HeNormal(1, fan_option='fan_out')
        with self.init_scope():
            self.conv1 = L.Convolution2D(256, 3, pad=1, initialW=initialW)
            self.conv2 = L.Convolution2D(256, 3, pad=1, initialW=initialW)
            self.conv3 = L.Convolution2D(256, 3, pad=1, initialW=initialW)
            self.conv4 = L.Convolution2D(256, 3, pad=1, initialW=initialW)
            self.conv5 = L.Deconvolution2D(
                256, 2, pad=0, stride=2, initialW=initialW)
            self.seg = L.Convolution2D(n_class, 1, pad=0, initialW=initialW)

        self._n_class = n_class
        self._scales = scales

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
            out_size = self.mask_size
            segs = chainer.Variable(
                self.xp.empty((0, self._n_class, out_size, out_size),
                              dtype=np.float32))
            return segs

        h = F.concat(pooled_hs, axis=0)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        return self.seg(h)

    def distribute(self, rois, roi_indices):
        size = self.xp.sqrt(
            self.xp.prod(rois[:, 2:] + 1 - rois[:, :2], axis=1))
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

    def decode(self, segms, bboxes, labels, sizes):
        # CPU is used because cv2.resize only accepts numpy arrays.
        segms = [chainer.backends.cuda.to_cpu(segm) for segm in segms]
        bboxes = [chainer.backends.cuda.to_cpu(bbox) for bbox in bboxes]
        labels = [chainer.backends.cuda.to_cpu(label) for label in labels]

        masks = []
        # To work around an issue with cv2.resize (it seems to automatically
        # pad with repeated border values), we manually zero-pad the masks by 1
        # pixel prior to resizing back to the original image resolution.
        # This prevents "top hat" artifacts. We therefore need to expand
        # the reference boxes by an appropriate factor.
        cv2_expand_scale = (self.mask_size + 2) / self.mask_size
        padded_mask = np.zeros((self.mask_size + 2, self.mask_size + 2),
                               dtype=np.float32)
        for bbox, segm, label, size in zip(
                bboxes, segms, labels, sizes):
            img_H, img_W = size
            mask = np.zeros((len(bbox), img_H, img_W), dtype=np.bool)

            bbox = expand_boxes(bbox, cv2_expand_scale)
            for i, (bb, sgm, lbl) in enumerate(zip(bbox, segm, label)):
                bb = bb.astype(np.int32)
                padded_mask[1:-1, 1:-1] = sgm[lbl + 1]

                # TODO(yuyu2172): Ignore +1 later
                bb_height = np.maximum(bb[2] - bb[0] + 1, 1)
                bb_width = np.maximum(bb[3] - bb[1] + 1, 1)

                crop_mask = cv2.resize(padded_mask, (bb_width, bb_height))
                crop_mask = crop_mask > 0.5

                y_min = max(bb[0], 0)
                x_min = max(bb[1], 0)
                y_max = min(bb[2] + 1, img_H)
                x_max = min(bb[3] + 1, img_W)
                mask[i, y_min:y_max, x_min:x_max] = crop_mask[
                    (y_min - bb[0]):(y_max - bb[0]),
                    (x_min - bb[1]):(x_max - bb[1])]
            masks.append(mask)
        return masks


def expand_boxes(bbox, scale):
    """Expand an array of boxes by a given scale."""
    xp = chainer.backends.cuda.get_array_module(bbox)

    h_half = (bbox[:, 2] - bbox[:, 0]) * .5
    w_half = (bbox[:, 3] - bbox[:, 1]) * .5
    y_c = (bbox[:, 2] + bbox[:, 0]) * .5
    x_c = (bbox[:, 3] + bbox[:, 1]) * .5

    h_half *= scale
    w_half *= scale

    expanded_bbox = xp.zeros(bbox.shape)
    expanded_bbox[:, 0] = y_c - h_half
    expanded_bbox[:, 1] = x_c - w_half
    expanded_bbox[:, 2] = y_c + h_half
    expanded_bbox[:, 3] = x_c + w_half

    return expanded_bbox


def mask_loss_pre(rois, roi_indices, gt_masks, gt_head_labels,
                  mask_size=28):
    xp = cuda.get_array_module(*rois)

    n_level = len(rois)

    roi_levels = xp.hstack(
        xp.array((l,) * len(rois[l])) for l in range(n_level)).astype(np.int32)
    rois = xp.vstack(rois).astype(np.float32)
    roi_indices = xp.hstack(roi_indices).astype(np.int32)
    gt_head_labels = xp.hstack(gt_head_labels)

    index = (gt_head_labels > 0).nonzero()[0]
    mask_roi_levels = roi_levels[index]
    mask_rois = rois[index]
    mask_roi_indices = roi_indices[index]
    gt_mask_labels = gt_head_labels[index]

    gt_segms = xp.empty((len(mask_rois), mask_size, mask_size), dtype=np.bool)
    for i in np.unique(cuda.to_cpu(mask_roi_indices)):
        gt_mask = gt_masks[i]
        gt_bbox = mask_to_bbox(gt_mask)

        index = (mask_roi_indices == i).nonzero()[0]
        mask_roi = mask_rois[index]
        iou = bbox_iou(mask_roi, gt_bbox)
        gt_index = iou.argmax(axis=1)
        gt_segms[index] = segm_wrt_bbox(
            gt_mask[gt_index], mask_roi, (M, M))

    # indices = [(mask_roi_levels == l).nonzero() for l in range(n_level)]
    flag_masks = [mask_roi_levels == l for l in range(n_level)]
    mask_rois = [mask_rois[m] for m in flag_masks]
    mask_roi_indices = [mask_roi_indices[m] for m in flag_masks]
    gt_segms = [gt_segms[m] for m in flag_masks]
    gt_mask_labels = [gt_mask_labels[m] for m in flag_masks]
    return mask_rois, mask_roi_indices, gt_segms, gt_mask_labels


def mask_loss_post(segms, mask_roi_indices, gt_segms, gt_mask_labels,
                   batchsize):
    # Just compute loss for the foreground class
    # divide by the batchsize
    xp = cuda.get_array_module(segms.array)

    mask_roi_indices = xp.hstack(mask_roi_indices).astype(np.int32)
    gt_segms = xp.vstack(gt_segms).astype(np.float32)
    gt_mask_labels = xp.hstack(gt_mask_labels).astype(np.int32)

    mask_loss = 0
    for i in np.unique(cuda.to_cpu(mask_roi_indices)):
        index = (mask_roi_indices == i).nonzero()[0]
        gt_segm = gt_segms[index]
        gt_mask_label = gt_mask_labels[index]

        mask_loss += F.sigmoid_cross_entropy(
            segms[index, gt_mask_label], gt_segm.astype(np.int32))

    mask_loss /= batchsize
    return mask_loss


def segm_wrt_bbox(mask, bbox, size):
    xp = chainer.backends.cuda.get_array_module(mask)

    bbox = bbox.astype(np.int32)

    segm = []
    for m, bb in zip(mask, bbox):
        if bb[2] - bb[0] == 0 or bb[3] - bb[1] == 0:
            segm.append(xp.zeros(size, dtype=np.bool))
            continue
        cropped_m = m[bb[0]:bb[2], bb[1]:bb[3]]
        cropped_m = chainer.backends.cuda.to_cpu(cropped_m)

        segm.append(resize(
            cropped_m[None].astype(np.float32),
            size, interpolation=PIL.Image.NEAREST)[0].astype(np.bool))
    return xp.array(segm, dtype=np.bool)
