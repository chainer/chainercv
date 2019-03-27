from __future__ import division

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links.model.fpn.misc import argsort
from chainercv.links.model.fpn.misc import choice
from chainercv.links.model.fpn.misc import exp_clip
from chainercv.links.model.fpn.misc import smooth_l1
from chainercv import utils


class RPN(chainer.Chain):
    """Region Proposal Network of Feature Pyramid Networks.

    Args:
        scales (tuple of floats): The scales of feature maps.

    """

    _anchor_size = 32
    _anchor_ratios = (0.5, 1, 2)
    _nms_thresh = 0.7
    _train_nms_limit_pre = 2000
    _train_nms_limit_post = 2000
    _test_nms_limit_pre = 1000
    _test_nms_limit_post = 1000

    def __init__(self, scales):
        super(RPN, self).__init__()

        init = {'initialW': initializers.Normal(0.01)}
        with self.init_scope():
            self.conv = L.Convolution2D(256, 3, pad=1, **init)
            self.loc = L.Convolution2D(len(self._anchor_ratios) * 4, 1, **init)
            self.conf = L.Convolution2D(len(self._anchor_ratios), 1, **init)

        self._scales = scales

    def forward(self, hs):
        """Calculates RoIs.

        Args:
            hs (iterable of array): An iterable of feature maps.

        Returns:
            tuple of two arrays:
            :obj:`locs` and :obj:`confs`.

            * **locs**: A list of arrays whose shape is \
                :math:`(N, K_l, 4)`, where :math:`N` is the size of batch and \
                :math:`K_l` is the number of the anchor boxes \
                of the :math:`l`-th level.
            " **confs**: A list of array whose shape is :math:`(N, K_l)`.
        """

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
        """Calculates anchor boxes.

        Args:
            sizes (iterable of tuples of two ints): An iterable of
                :math:`(H_l, W_l)`, where :math:`H_l` and :math:`W_l`
                are height and width of the :math:`l`-th feature map.

        Returns:
            list of arrays:
            The shape of the :math:`l`-th array is :math:`(H_l * W_l * A, 4)`,
            where :math:`A` is the number of anchor ratios.

        """
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
        """Decodes back to coordinates of RoIs.

        This method decodes :obj:`locs` and :obj:`confs` returned
        by a FPN network back to :obj:`rois` and :obj:`roi_indices`.

        Args:
            locs (list of arrays): A list of arrays whose shape is
                :math:`(N, K_l, 4)`, where :math:`N` is the size of batch and
                :math:`N_l` is the number of the anchor boxes
                of the :math:`l`-th level.
            confs (list of arrays): A list of array whose shape is
                :math:`(N, K_l)`.
            anchors (list of arrays): Anchor boxes returned by :meth:`anchors`.
            in_shape (tuple of ints): The shape of input of array
                the feature extractor.

        Returns:
            tuple of two arrays:
            :obj:`rois` and :obj:`roi_indices`.

            * **rois**: An array of shape :math:`(R, 4)`, \
                where :math:`R` is the total number of RoIs in the given batch.
            * **roi_indices** : An array of shape :math:`(R,)`.
        """

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

                order = argsort(-conf_l)[:nms_limit_pre]
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

            order = argsort(-conf)[:nms_limit_post]
            roi = roi[order]

            rois.append(roi)
            roi_indices.append(self.xp.array((i,) * len(roi)))

        rois = self.xp.vstack(rois).astype(np.float32)
        roi_indices = self.xp.hstack(roi_indices).astype(np.int32)
        return rois, roi_indices


def rpn_loss(locs, confs, anchors, sizes,  bboxes):
    """Loss function for RPN.

     Args:
         locs (iterable of arrays): An iterable of arrays whose shape is
             :math:`(N, K_l, 4)`, where :math:`K_l` is the number of
             the anchor boxes of the :math:`l`-th level.
         confs (iterable of arrays): An iterable of arrays whose shape is
             :math:`(N, K_l)`.
         anchors (list of arrays): A list of arrays returned by
             :meth:`anchors`.
         sizes (list of tuples of two ints): A list of
             :math:`(H_n, W_n)`, where :math:`H_n` and :math:`W_n`
             are height and width of the :math:`n`-th image.
         bboxes (list of arrays): A list of arrays whose shape is
             :math:`(R_n, 4)`, where :math:`R_n` is the number of
             ground truth bounding boxes.

     Returns:
         tuple of two variables:
         :obj:`loc_loss` and :obj:`conf_loss`.
    """
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
            gt_label[choice(fg_index, size=len(fg_index) - n_fg)] = -1

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
