from __future__ import division

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.functions import ps_roi_max_align_2d
from chainercv.links.model.fpn.misc import argsort
from chainercv.links.model.fpn.misc import choice
from chainercv.links.model.fpn.misc import exp_clip
from chainercv.links.model.fpn.misc import smooth_l1
from chainercv.links.model.light_head_rcnn.global_context_module import \
    GlobalContextModule
from chainercv import utils


class BboxHeadBase(chainer.Chain):
    def distribute(self, rois, roi_indices):
        """Assigns Rois to feature maps according to their size.

        Args:
            rois (array): An array of shape :math:`(R, 4)`, \
                where :math:`R` is the total number of RoIs in the given batch.
            roi_indices (array): An array of shape :math:`(R,)`.

        Returns:
            tuple of two lists:
            :obj:`rois` and :obj:`roi_indices`.

            * **rois**: A list of arrays of shape :math:`(R_l, 4)`, \
                where :math:`R_l` is the number of RoIs in the :math:`l`-th \
                feature map.
            * **roi_indices** : A list of arrays of shape :math:`(R_l,)`.
        """

        size = self.xp.sqrt(self.xp.prod(rois[:, 2:] - rois[:, :2], axis=1))
        level = self.xp.floor(self.xp.log2(
            size / self._canonical_scale + 1e-6)).astype(np.int32)
        # skip last level
        level = self.xp.clip(
            level + self._canonical_level, 0, len(self._scales) - 2)

        masks = [level == l for l in range(len(self._scales))]
        rois = [rois[mask] for mask in masks]
        roi_indices = [roi_indices[mask] for mask in masks]

        return rois, roi_indices

    def decode(self, rois, roi_indices, locs, confs,
               scales, sizes, nms_thresh, score_thresh):
        """Decodes back to coordinates of RoIs.

        This method decodes :obj:`locs` and :obj:`confs` returned
        by a FPN network back to :obj:`bboxes`,
        :obj:`labels` and :obj:`scores`.

        Args:
            rois (iterable of arrays): An iterable of arrays of
                shape :math:`(R_l, 4)`, where :math:`R_l` is the number
                of RoIs in the :math:`l`-th feature map.
            roi_indices (iterable of arrays): An iterable of arrays of
                shape :math:`(R_l,)`.
            locs (array): An array whose shape is :math:`(R, n\_class, 4)`,
                where :math:`R` is the total number of RoIs in the given batch.
            confs (array): An array whose shape is :math:`(R, n\_class)`.
            scales (list of floats): A list of floats returned
                by :meth:`~chainercv.links.model.fpn.faster_rcnn.prepare`
            sizes (list of tuples of two ints): A list of
                :math:`(H_n, W_n)`, where :math:`H_n` and :math:`W_n`
                are height and width of the :math:`n`-th image.
            nms_thresh (float): The threshold value
                for :func:`~chainercv.utils.non_maximum_suppression`.
            score_thresh (float): The threshold value for confidence score.

        Returns:
            tuple of three list of arrays:
            :obj:`bboxes`, :obj:`labels` and :obj:`scores`.

           * **bboxes**: A list of float arrays of shape :math:`(R'_n, 4)`, \
               where :math:`R'_n` is the number of bounding boxes in \
               the :math:`n`-th image. \
               Each bounding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R'_n,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R'_n,)`. \
               Each value indicates how confident the prediction is.
        """

        rois = self.xp.vstack(rois)
        roi_indices = self.xp.hstack(roi_indices)
        locs = locs.array
        confs = confs.array

        bboxes = []
        labels = []
        scores = []
        for i in range(len(scales)):
            mask = roi_indices == i
            roi = rois[mask]
            loc = locs[mask]
            conf = confs[mask]

            bbox = self.xp.broadcast_to(roi[:, None], loc.shape) / scales[i]
            # tlbr -> yxhw
            bbox[:, :, 2:] -= bbox[:, :, :2]
            bbox[:, :, :2] += bbox[:, :, 2:] / 2
            # offset
            bbox[:, :, :2] += loc[:, :, :2] * bbox[:, :, 2:] * self.std[0]
            bbox[:, :, 2:] *= self.xp.exp(
                self.xp.minimum(loc[:, :, 2:] * self.std[1], exp_clip))
            # yxhw -> tlbr
            bbox[:, :, :2] -= bbox[:, :, 2:] / 2
            bbox[:, :, 2:] += bbox[:, :, :2]
            # clip
            bbox[:, :, :2] = self.xp.maximum(bbox[:, :, :2], 0)
            bbox[:, :, 2:] = self.xp.minimum(
                bbox[:, :, 2:], self.xp.array(sizes[i]))

            conf = self.xp.exp(conf)
            score = conf / self.xp.sum(conf, axis=1, keepdims=True)

            bbox, label, score = _suppress(
                bbox, score, nms_thresh, score_thresh)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores


class BboxHead(BboxHeadBase):
    """Bounding box head network of Feature Pyramid Networks.

    Args:
        n_class (int): The number of classes including background.
        scales (tuple of floats): The scales of feature maps.

    """
    _canonical_level = 2
    _canonical_scale = 224
    _roi_size = 7
    _roi_sample_ratio = 2
    std = (0.1, 0.2)

    def __init__(self, n_class, scales):
        super(BboxHead, self).__init__()

        fc_init = {
            'initialW': Caffe2FCUniform(),
            'initial_bias': Caffe2FCUniform(),
        }
        with self.init_scope():
            self.fc1 = L.Linear(1024, **fc_init)
            self.fc2 = L.Linear(1024, **fc_init)
            self.loc = L.Linear(
                n_class * 4, initialW=initializers.Normal(0.001))
            self.conf = L.Linear(n_class, initialW=initializers.Normal(0.01))

        self._n_class = n_class
        self._scales = scales

    def forward(self, hs, rois, roi_indices):
        """Calculates RoIs.

        Args:
            hs (iterable of array): An iterable of feature maps.
            rois (list of arrays): A list of arrays of shape: math: `(R_l, 4)`,
                where: math: `R_l` is the number of RoIs in the: math: `l`- th
                feature map.
            roi_indices (list of arrays): A list of arrays of
                shape :math:`(R_l,)`.

        Returns:
            tuple of two arrays:
            :obj:`locs` and :obj:`confs`.

            * **locs**: An arrays whose shape is \
                :math:`(R, n\_class, 4)`, where :math:`R` is the total number \
                of RoIs in the batch.
            * **confs**: A list of array whose shape is :math:`(R, n\_class)`.
        """

        hs_ = []
        for l, h in enumerate(hs):
            if len(rois[l]) == 0:
                continue
            h = F.roi_average_align_2d(
                h, rois[l], roi_indices[l], self._roi_size,
                self._scales[l], self._roi_sample_ratio)
            hs_.append(h)
        hs = hs_

        if len(hs) == 0:
            locs = chainer.Variable(
                self.xp.empty((0, self._n_class, 4), dtype=np.float32))
            confs = chainer.Variable(
                self.xp.empty((0, self._n_class), dtype=np.float32))
            return locs, confs

        h = F.concat(hs, axis=0)
        h = F.reshape(h, (h.shape[0], -1))
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        locs = self.loc(h)
        locs = F.reshape(locs, (locs.shape[0], -1, 4))
        confs = self.conf(h)
        return locs, confs


class LightBboxHead(BboxHeadBase):
    """Bounding box light head network of Feature Pyramid Networks.

    Args:
        n_class (int): The number of classes including background.
        scales (tuple of floats): The scales of feature maps.

    """
    _canonical_level = 2
    _canonical_scale = 224
    _roi_size = 7
    _roi_sample_ratio = 2
    std = (0.1, 0.2)

    def __init__(self, n_class, scales):
        super(LightBboxHead, self).__init__()

        with self.init_scope():
            self.global_context_module = GlobalContextModule(
                2048, 256, self._roi_size * self._roi_size * 10, 15,
                initialW=chainer.initializers.Normal(0.01))
            self.fc1 = L.Linear(
                2048, initialW=chainer.initializers.Normal(0.01))
            self.loc = L.Linear(
                n_class * 4, initialW=initializers.Normal(0.001))
            self.conf = L.Linear(n_class, initialW=initializers.Normal(0.01))

        self._n_class = n_class
        self._scales = scales

    def forward(self, hs, rois, roi_indices):
        """Calculates RoIs.

        Args:
            hs (iterable of array): An iterable of feature maps.
            rois (list of arrays): A list of arrays of shape: math: `(R_l, 4)`,
                where: math: `R_l` is the number of RoIs in the: math: `l`- th
                feature map.
            roi_indices (list of arrays): A list of arrays of
                shape :math:`(R_l,)`.

        Returns:
            tuple of two arrays:
            :obj:`locs` and :obj:`confs`.

            * **locs**: An arrays whose shape is \
                :math:`(R, n\_class, 4)`, where :math:`R` is the total number \
                of RoIs in the batch.
            * **confs**: A list of array whose shape is :math:`(R, n\_class)`.
        """

        hs_ = []
        for l, h in enumerate(hs):
            if len(rois[l]) == 0:
                continue
            h = self.global_context_module(h)
            h = ps_roi_max_align_2d(
                h, rois[l], roi_indices[l],
                (10, self._roi_size, self._roi_size),
                self._scales[l], self._roi_size,
                self._roi_sample_ratio)
            h = F.where(
                self.xp.isinf(h.array),
                self.xp.zeros(h.shape, dtype=h.dtype), h)
            hs_.append(h)
        hs = hs_

        if len(hs) == 0:
            locs = chainer.Variable(
                self.xp.empty((0, self._n_class, 4), dtype=np.float32))
            confs = chainer.Variable(
                self.xp.empty((0, self._n_class), dtype=np.float32))
            return locs, confs

        h = F.concat(hs, axis=0)
        h = F.reshape(h, (h.shape[0], -1))
        h = F.relu(self.fc1(h))

        locs = self.loc(h)
        locs = F.reshape(locs, (locs.shape[0], -1, 4))
        confs = self.conf(h)
        return locs, confs


def bbox_head_loss_pre(rois, roi_indices, std, bboxes, labels):
    """Loss function for Head (pre).

    This function processes RoIs for :func:`bbox_head_loss_post`.

    Args:
        rois (iterable of arrays): An iterable of arrays of
            shape :math:`(R_l, 4)`, where :math:`R_l` is the number
            of RoIs in the :math:`l`-th feature map.
        roi_indices (iterable of arrays): An iterable of arrays of
            shape :math:`(R_l,)`.
        std (tuple of floats): Two coefficients used for encoding
            bounding boxes.
        bboxes (list of arrays): A list of arrays whose shape is
            :math:`(R_n, 4)`, where :math:`R_n` is the number of
            ground truth bounding boxes.
        labels (list of arrays): A list of arrays whose shape is
            :math:`(R_n,)`.

     Returns:
         tuple of four lists:
         :obj:`rois`, :obj:`roi_indices`, :obj:`gt_locs`, and :obj:`gt_labels`.

          * **rois**: A list of arrays of shape :math:`(R'_l, 4)`, \
              where :math:`R'_l` is the number of RoIs in the :math:`l`-th \
              feature map.
          * **roi_indices**: A list of arrays of shape :math:`(R'_l,)`.
          * **gt_locs**: A list of arrays of shape :math:`(R'_l, 4) \
              indicating the bounding boxes of ground truth.
          * **roi_indices**: A list of arrays of shape :math:`(R'_l,)` \
              indicating the classes of ground truth.
    """

    thresh = 0.5
    batchsize_per_image = 512
    fg_ratio = 0.25

    xp = cuda.get_array_module(*rois)

    n_level = len(rois)
    roi_levels = xp.hstack(
        xp.array((l,) * len(rois[l])) for l in range(n_level)).astype(np.int32)
    rois = xp.vstack(rois).astype(np.float32)
    roi_indices = xp.hstack(roi_indices).astype(np.int32)

    rois_yx = (rois[:, 2:] + rois[:, :2]) / 2
    rois_hw = rois[:, 2:] - rois[:, :2]
    indices = np.unique(cuda.to_cpu(roi_indices))

    gt_locs = xp.empty_like(rois)
    gt_labels = xp.empty_like(roi_indices)
    for i in indices:
        mask = roi_indices == i

        if len(bboxes[i]) > 0:
            iou = utils.bbox_iou(rois[mask], bboxes[i])
            gt_index = iou.argmax(axis=1)

            gt_loc = bboxes[i][gt_index].copy()
        else:
            gt_loc = xp.empty_like(rois[mask])
        # tlbr -> yxhw
        gt_loc[:, 2:] -= gt_loc[:, :2]
        gt_loc[:, :2] += gt_loc[:, 2:] / 2
        # offset
        gt_loc[:, :2] = (gt_loc[:, :2] - rois_yx[mask]) / \
            rois_hw[mask] / std[0]
        gt_loc[:, 2:] = xp.log(gt_loc[:, 2:] / rois_hw[mask]) / std[1]

        if len(bboxes[i]) > 0:
            gt_label = labels[i][gt_index] + 1
            gt_label[iou.max(axis=1) < thresh] = 0
        else:
            gt_label = xp.zeros(int(mask.sum()), dtype=np.int32)

        fg_index = xp.where(gt_label > 0)[0]
        n_fg = int(batchsize_per_image * fg_ratio)
        if len(fg_index) > n_fg:
            gt_label[choice(fg_index, size=len(fg_index) - n_fg)] = -1

        bg_index = xp.where(gt_label == 0)[0]
        n_bg = batchsize_per_image - int((gt_label > 0).sum())
        if len(bg_index) > n_bg:
            gt_label[choice(bg_index, size=len(bg_index) - n_bg)] = -1

        gt_locs[mask] = gt_loc
        gt_labels[mask] = gt_label

    mask = gt_labels >= 0
    rois = rois[mask]
    roi_indices = roi_indices[mask]
    roi_levels = roi_levels[mask]
    gt_locs = gt_locs[mask]
    gt_labels = gt_labels[mask]

    masks = [roi_levels == l for l in range(n_level)]
    rois = [rois[m] for m in masks]
    roi_indices = [roi_indices[m] for m in masks]
    gt_locs = [gt_locs[m] for m in masks]
    gt_labels = [gt_labels[m] for m in masks]

    return rois, roi_indices, gt_locs, gt_labels


def bbox_head_loss_post(
        locs, confs, roi_indices, gt_locs, gt_labels, batchsize):
    """Loss function for Head (post).

     Args:
         locs (array): An array whose shape is :math:`(R, n\_class, 4)`,
             where :math:`R` is the total number of RoIs in the given batch.
         confs (array): An iterable of arrays whose shape is
             :math:`(R, n\_class)`.
         roi_indices (list of arrays): A list of arrays returned by
             :func:`bbox_head_locs_pre`.
         gt_locs (list of arrays): A list of arrays returned by
             :func:`bbox_head_locs_pre`.
         gt_labels (list of arrays): A list of arrays returned by
             :func:`bbox_head_locs_pre`.
         batchsize (int): The size of batch.

     Returns:
         tuple of two variables:
         :obj:`loc_loss` and :obj:`conf_loss`.
    """

    xp = cuda.get_array_module(locs.array, confs.array)

    roi_indices = xp.hstack(roi_indices).astype(np.int32)
    gt_locs = xp.vstack(gt_locs).astype(np.float32)
    gt_labels = xp.hstack(gt_labels).astype(np.int32)

    loc_loss = 0
    conf_loss = 0
    for i in np.unique(cuda.to_cpu(roi_indices)):
        mask = roi_indices == i
        gt_loc = gt_locs[mask]
        gt_label = gt_labels[mask]

        n_sample = mask.sum()
        loc_loss += F.sum(smooth_l1(
            locs[mask][xp.where(gt_label > 0)[0], gt_label[gt_label > 0]],
            gt_loc[gt_label > 0], 1)) / n_sample
        conf_loss += F.softmax_cross_entropy(confs[mask], gt_label)

    loc_loss /= batchsize
    conf_loss /= batchsize

    return loc_loss, conf_loss


class Caffe2FCUniform(chainer.initializer.Initializer):
    """Initializer used in Caffe2.

    """

    def __call__(self, array):
        scale = 1 / np.sqrt(array.shape[-1])
        initializers.Uniform(scale)(array)


def _suppress(raw_bbox, raw_score, nms_thresh, score_thresh):
    xp = cuda.get_array_module(raw_bbox, raw_score)

    bbox = []
    label = []
    score = []
    for l in range(raw_score.shape[1] - 1):
        bbox_l = raw_bbox[:, l + 1]
        score_l = raw_score[:, l + 1]

        mask = score_l >= score_thresh
        bbox_l = bbox_l[mask]
        score_l = score_l[mask]

        order = argsort(-score_l)
        bbox_l = bbox_l[order]
        score_l = score_l[order]
        indices = utils.non_maximum_suppression(bbox_l, nms_thresh)
        bbox_l = bbox_l[indices]
        score_l = score_l[indices]

        bbox.append(bbox_l)
        label.append(xp.array((l,) * len(bbox_l)))
        score.append(score_l)

    bbox = xp.vstack(bbox).astype(np.float32)
    label = xp.hstack(label).astype(np.int32)
    score = xp.hstack(score).astype(np.float32)
    return bbox, label, score
