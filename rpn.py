import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv import utils

from fpn import exp_clip
from fpn.smooth_l1 import smooth_l1


class RPN(chainer.Chain):

    _anchor_size = 32
    _anchor_ratios = (0.5, 1, 2)
    _nms_thresh = 0.7
    _train_nms_limit_pre = 2000
    _train_nms_limit_post = 2000
    _test_nms_limit_pre = 1000
    _test_nms_limit_post = 1000

    def __init__(self, scales):
        super().__init__()

        init = {'initialW': initializers.Normal(0.01)}
        with self.init_scope():
            self.conv = L.Convolution2D(256, 3, pad=1, **init)
            self.loc = L.Convolution2D(len(self._anchor_ratios) * 4, 1, **init)
            self.conf = L.Convolution2D(len(self._anchor_ratios), 1, **init)

        self._scales = scales

    def __call__(self, hs):
        locs = []
        confs = []
        for h in hs:
            h = F.relu(self.conv(h))

            loc = self.loc(h)
            loc = F.transpose(loc, (0, 2, 3, 1))
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            locs.append(loc)

            conf = self.conf(h)
            conf = F.transpose(conf, (0, 2, 3, 1))
            conf = F.reshape(conf, (conf.shape[0], -1))
            confs.append(conf)

        return locs, confs

    def anchors(self, sizes):
        anchors = []
        for l, (H, W) in enumerate(sizes):
            v, u, ar = np.meshgrid(
                np.arange(W), np.arange(H), self._anchor_ratios)
            w = np.round(1 / np.sqrt(ar) / self._scales[l])
            h = np.round(w * ar)
            anchor = np.stack((u, v, h, w)).reshape((4, -1)).transpose()
            anchor[:, :2] = (anchor[:, :2] + 0.5) / self._scales[l]
            anchor[:, 2:] *= (self._anchor_size << l) * self._scales[l]
            # yxhw -> tlbr
            anchor[:, :2] -= anchor[:, 2:] / 2
            anchor[:, 2:] += anchor[:, :2]
            anchors.append(self.xp.array(anchor, dtype=np.float32))

        return anchors

    def decode(self, locs, confs, anchors, in_shape):
        if chainer.config.train:
            nms_limit_pre = self._train_nms_limit_pre
            nms_limit_post = self._train_nms_limit_post
        else:
            nms_limit_pre = self._test_nms_limit_pre
            nms_limit_post = self._test_nms_limit_post

        rois = []
        roi_indices = []
        for i in range(in_shape[0]):
            roi = []
            conf = []
            for l in range(len(self._scales)):
                loc_l = locs[l].array[i]
                conf_l = confs[l].array[i]

                roi_l = anchors[l].copy()
                # tlbr -> yxhw
                roi_l[:, 2:] -= roi_l[:, :2]
                roi_l[:, :2] += roi_l[:, 2:] / 2
                # offset
                roi_l[:, :2] += loc_l[:, :2] * roi_l[:, 2:]
                roi_l[:, 2:] *= self.xp.exp(
                    self.xp.minimum(loc_l[:, 2:], exp_clip))
                # yxhw -> tlbr
                roi_l[:, :2] -= roi_l[:, 2:] / 2
                roi_l[:, 2:] += roi_l[:, :2]
                # clip
                roi_l[:, :2] = self.xp.maximum(roi_l[:, :2], 0)
                roi_l[:, 2:] = self.xp.minimum(
                    roi_l[:, 2:], self.xp.array(in_shape[2:]))

                order = _argsort(-conf_l)[:nms_limit_pre]
                roi_l = roi_l[order]
                conf_l = conf_l[order]

                mask = (roi_l[:, 2:] - roi_l[:, :2] > 0).all(axis=1)
                roi_l = roi_l[mask]
                conf_l = conf_l[mask]

                indices = utils.non_maximum_suppression(
                    roi_l, self._nms_thresh, limit=nms_limit_post)
                roi_l = roi_l[indices]
                conf_l = conf_l[indices]

                roi.append(roi_l)
                conf.append(conf_l)

            roi = self.xp.vstack(roi).astype(np.float32)
            conf = self.xp.hstack(conf).astype(np.float32)

            order = _argsort(-conf)[:nms_limit_post]
            roi = roi[order]

            rois.append(roi)
            roi_indices.append(self.xp.array((i,) * len(roi)))

        rois = self.xp.vstack(rois).astype(np.float32)
        roi_indices = self.xp.hstack(roi_indices).astype(np.int32)
        return rois, roi_indices


def rpn_loss(locs, confs, anchors, sizes,  bboxes):
    fg_thresh = 0.7
    bg_thresh = 0.3
    batchsize_per_image = 256
    fg_ratio = 0.25

    locs = F.concat(locs)
    confs = F.concat(confs)

    xp = cuda.get_array_module(locs.array, confs.array)

    anchors = xp.vstack(anchors)
    anchors_yx = (anchors[:, 2:] + anchors[:, :2]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]

    loc_loss = 0
    conf_loss = 0
    for i in range(len(sizes)):
        if len(bboxes[i]) > 0:
            iou = utils.bbox_iou(anchors, bboxes[i])

            gt_loc = bboxes[i][iou.argmax(axis=1)].copy()
            # tlbr -> yxhw
            gt_loc[:, 2:] -= gt_loc[:, :2]
            gt_loc[:, :2] += gt_loc[:, 2:] / 2
            # offset
            gt_loc[:, :2] = (gt_loc[:, :2] - anchors_yx) / anchors_hw
            gt_loc[:, 2:] = xp.log(gt_loc[:, 2:] / anchors_hw)
        else:
            gt_loc = xp.empty_like(anchors)

        gt_label = xp.empty(len(anchors), dtype=np.int32)
        gt_label[:] = -1

        mask = xp.logical_and(
            anchors[:, :2] >= 0,
            anchors[:, 2:] < xp.array(sizes[i])).all(axis=1)

        if len(bboxes[i]) > 0:
            gt_label[xp.where(mask)[0]
                     [(iou[mask] == iou[mask].max(axis=0)).any(axis=1)]] = 1
            gt_label[xp.logical_and(mask, iou.max(axis=1) >= fg_thresh)] = 1

        fg_index = xp.where(gt_label == 1)[0]
        n_fg = int(batchsize_per_image * fg_ratio)
        if len(fg_index) > n_fg:
            gt_label[_choice(fg_index, size=len(fg_index) - n_fg)] = -1

        if len(bboxes[i]) > 0:
            bg_index = xp.where(xp.logical_and(
                mask, iou.max(axis=1) < bg_thresh))[0]
        else:
            bg_index = xp.where(mask)[0]
        n_bg = batchsize_per_image - int((gt_label == 1).sum())
        if len(bg_index) > n_bg:
            gt_label[bg_index[
                xp.random.randint(len(bg_index), size=n_bg)]] = 0

        n_sample = (gt_label >= 0).sum()
        loc_loss += F.sum(smooth_l1(
            locs[i][gt_label == 1], gt_loc[gt_label == 1], 1 / 9)) / n_sample
        conf_loss += F.sum(F.sigmoid_cross_entropy(
            confs[i][gt_label >= 0], gt_label[gt_label >= 0], reduce='no')) \
            / n_sample

    loc_loss /= len(sizes)
    conf_loss /= len(sizes)

    return loc_loss, conf_loss


# to avoid out of memory
def _argsort(x):
    xp = cuda.get_array_module(x)
    i = np.argsort(cuda.to_cpu(x))
    if xp is np:
        return i
    else:
        return cuda.to_gpu(i)


# to avoid out of memory
def _choice(x, size):
    xp = cuda.get_array_module(x)
    y = np.random.choice(cuda.to_cpu(x), size, replace=False)
    if xp is np:
        return y
    else:
        return cuda.to_gpu(y)
