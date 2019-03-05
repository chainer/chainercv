from __future__ import division

import numpy as np
import PIL

import cv2

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer.initializers import HeNormal
import chainer.links as L

from chainercv.links import Conv2DActiv
from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou

from chainercv.links.model.mask_rcnn.misc import segm_to_mask
from chainercv.links.model.mask_rcnn.misc import mask_to_segm


class MaskHead(chainer.Chain):

    """Mask Head network of Mask R-CNN.

    Args:
        n_class (int): The number of classes including background.
        scales (tuple of floats): The scales of feature maps.

    """

    _canonical_level = 2
    _canonical_scale = 224
    _roi_size = 14
    _roi_sample_ratio = 2
    segm_size = _roi_size * 2

    def __init__(self, n_class, scales):
        super(MaskHead, self).__init__()

        initialW = HeNormal(1, fan_option='fan_out')
        with self.init_scope():
            self.conv1 = Conv2DActiv(256, 3, pad=1, initialW=initialW)
            self.conv2 = Conv2DActiv(256, 3, pad=1, initialW=initialW)
            self.conv3 = Conv2DActiv(256, 3, pad=1, initialW=initialW)
            self.conv4 = Conv2DActiv(256, 3, pad=1, initialW=initialW)
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
            out_size = self.segm_size
            segs = chainer.Variable(
                self.xp.empty((0, self._n_class, out_size, out_size),
                              dtype=np.float32))
            return segs

        h = F.concat(pooled_hs, axis=0)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = F.relu(self.conv5(h))
        return self.seg(h)

    def distribute(self, rois, roi_indices):
        """Assigns feature levels to Rois based on their size.

        Args:
            rois (array): An array of shape :math:`(R, 4)`, \
                where :math:`R` is the total number of RoIs in the given batch.
            roi_indices (array): An array of shape :math:`(R,)`.

        Returns:
            two lists and one array:
            :obj:`out_rois`, :obj:`out_roi_indices` and :obj:`order`.

            * **out_rois**: A list of arrays of shape :math:`(R_l, 4)`, \
                where :math:`R_l` is the number of RoIs in the :math:`l`-th \
                feature map.
            * **out_roi_indices** : A list of arrays of shape :math:`(R_l,)`.
            * **order**: A correspondence between the output and the input. \
                The relationship below is satisfied.

            .. code:: python

                xp.concatenate(out_rois, axis=0)[order[i]] == rois[i]

        """

        size = self.xp.sqrt(self.xp.prod(rois[:, 2:] - rois[:, :2], axis=1))
        level = self.xp.floor(self.xp.log2(
            size / self._canonical_scale + 1e-6)).astype(np.int32)
        # skip last level
        level = self.xp.clip(
            level + self._canonical_level, 0, len(self._scales) - 2)

        masks = [level == l for l in range(len(self._scales))]
        out_rois = [rois[mask] for mask in masks]
        out_roi_indices = [roi_indices[mask] for mask in masks]
        order = self.xp.argsort(
            self.xp.concatenate([self.xp.where(mask)[0] for mask in masks]))
        return out_rois, out_roi_indices, order

    def decode(self, segms, bboxes, labels, sizes):
        """Decodes back to masks.

        Args:
            segms (iterable of arrays): An iterable of arrays of
                shape :math:`(R_n, n\_class, M, M)`.
            bboxes (iterable of arrays): An iterable of arrays of
                shape :math:`(R_n, 4)`.
            labels (iterable of arrays): An iterable of arrays of
                shape :math:`(R_n,)`.
            sizes (list of tuples of two ints): A list of
                :math:`(H_n, W_n)`, where :math:`H_n` and :math:`W_n`
                are height and width of the :math:`n`-th image.

        Returns:
            list of arrays:
            This list contains instance segmentation for each image
            in the batch.
            More precisely, this is a list of boolean arrays of shape
            :math:`(R'_n, H_n, W_n)`, where :math:`R'_n` is the number of
            bounding boxes in the :math:`n`-th image.
        """

        xp = chainer.backends.cuda.get_array_module(*segms)
        if xp != np:
            raise ValueError(
                'MaskHead.decode only supports numpy inputs for now.')
        masks = []
        for bbox, segm, label, size in zip(
                bboxes, segms, labels, sizes):
            masks.append(
                segm_to_mask(segm[np.arange(len(label)), label + 1],
                             bbox, size))
        return masks


def mask_loss_pre(rois, roi_indices, gt_masks, gt_bboxes,
                  gt_head_labels, segm_size):
    """Loss function for Mask Head (pre).

    This function processes RoIs for :func:`mask_loss_post` by
    selecting RoIs for mask loss calculation and
    preparing ground truth network output.

    Args:
        rois (iterable of arrays): An iterable of arrays of
            shape :math:`(R_l, 4)`, where :math:`R_l` is the number
            of RoIs in the :math:`l`-th feature map.
        roi_indices (iterable of arrays): An iterable of arrays of
            shape :math:`(R_l,)`.
        gt_masks (iterable of arrays): An iterable of arrays whose shape is
            :math:`(R_n, H, W)`, where :math:`R_n` is the number of
            ground truth objects.
        gt_head_labels (iterable of arrays): An iterable of arrays of
            shape :math:`(R_l,)`. This is a collection of ground-truth
            labels assigned to :obj:`rois` during bounding box localization
            stage. The range of value is :math:`(0, n\_class - 1)`.
        segm_size (int): Size of the ground truth network output.

    Returns:
        tuple of four lists:
        :obj:`mask_rois`, :obj:`mask_roi_indices`,
        :obj:`gt_segms`, and :obj:`gt_mask_labels`.

        * **rois**: A list of arrays of shape :math:`(R'_l, 4)`, \
            where :math:`R'_l` is the number of RoIs in the :math:`l`-th \
            feature map.
        * **roi_indices**: A list of arrays of shape :math:`(R'_l,)`.
        * **gt_segms**: A list of arrays of shape :math:`(R'_l, M, M). \
            :math:`M` is the argument :obj:`segm_size`.
        * **gt_mask_labels**: A list of arrays of shape :math:`(R'_l,)` \
            indicating the classes of ground truth.
    """

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

    gt_segms = xp.empty((len(mask_rois), segm_size, segm_size), dtype=np.bool)
    for i in np.unique(cuda.to_cpu(mask_roi_indices)):
        gt_mask = gt_masks[i]
        gt_bbox = gt_bboxes[i]

        index = (mask_roi_indices == i).nonzero()[0]
        mask_roi = mask_rois[index]
        iou = bbox_iou(mask_roi, gt_bbox)
        gt_index = iou.argmax(axis=1)
        gt_segms[index] = xp.array(
            mask_to_segm(gt_mask, mask_roi, segm_size, gt_index))

    flag_masks = [mask_roi_levels == l for l in range(n_level)]
    mask_rois = [mask_rois[m] for m in flag_masks]
    mask_roi_indices = [mask_roi_indices[m] for m in flag_masks]
    gt_segms = [gt_segms[m] for m in flag_masks]
    gt_mask_labels = [gt_mask_labels[m] for m in flag_masks]
    return mask_rois, mask_roi_indices, gt_segms, gt_mask_labels


def mask_loss_post(segms, mask_roi_indices, gt_segms, gt_mask_labels,
                   batchsize):
    """Loss function for Head (post).

     Args:
         segms (array): An array whose shape is :math:`(R, n\_class, M, M)`,
             where :math:`R` is the total number of RoIs in the given batch.
         mask_roi_indices (array): A list of arrays returned by
             :func:`mask_loss_pre`.
         gt_segms (list of arrays): A list of arrays returned by
             :func:`mask_loss_pre`.
         gt_mask_labels (list of arrays): A list of arrays returned by
             :func:`mask_loss_pre`.
         batchsize (int): The size of batch.

     Returns:
        chainer.Variable:
        Mask loss.
    """
    xp = cuda.get_array_module(segms.array)

    mask_roi_indices = xp.hstack(mask_roi_indices).astype(np.int32)
    gt_segms = xp.vstack(gt_segms).astype(np.float32, copy=False)
    gt_mask_labels = xp.hstack(gt_mask_labels).astype(np.int32)

    mask_loss = F.sigmoid_cross_entropy(
        segms[np.arange(len(gt_mask_labels)), gt_mask_labels],
        gt_segms.astype(np.int32))
    return mask_loss


def _segm_wrt_bbox(mask, gt_index, bbox, size, xp):
    bbox = chainer.backends.cuda.to_cpu(bbox.astype(np.int32))

    segm = []
    for i, bb in zip(chainer.backends.cuda.to_cpu(gt_index), bbox):
        cropped_m = mask[i, bb[0]:bb[2], bb[1]:bb[3]]
        cropped_m = chainer.backends.cuda.to_cpu(cropped_m)
        if cropped_m.shape[0] == 0 or cropped_m.shape[1] == 0:
            segm.append(np.zeros(size, dtype=np.bool))
            continue

        segm.append(resize(
            cropped_m[None].astype(np.float32),
            size, interpolation=PIL.Image.NEAREST)[0])
    return xp.array(segm, dtype=np.float32)
