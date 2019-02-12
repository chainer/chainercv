from __future__ import division

import numpy as np
import PIL

import chainer
from chainer.backends import cuda
import chainer.functions as F

from chainercv import transforms


class MaskRCNN(chainer.Chain):

    _min_size = 800
    _max_size = 1333
    _stride = 32

    def __init__(self, extractor, rpn, head, mask_head):
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head
            self.mask_head = mask_head

        self.use_preset('visualize')

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.5
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def __call__(self, x):
        assert(not chainer.config.train)
        hs = self.extractor(x)
        rpn_locs, rpn_confs = self.rpn(hs)
        anchors = self.rpn.anchors(h.shape[2:] for h in hs)
        rois, roi_indices = self.rpn.decode(
            rpn_locs, rpn_confs, anchors, x.shape)
        rois, roi_indices = self.head.distribute(rois, roi_indices)
        return hs, rois, roi_indices

    def predict(self, imgs):
        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            hs, rois, roi_indices = self(x)
            head_locs, head_confs = self.head(hs, rois, roi_indices)
        bboxes, labels, scores = self.head.decode(
            rois, roi_indices, head_locs, head_confs,
            scales, sizes, self.nms_thresh, self.score_thresh)

        # Rescale bbox to the scaled resolution
        rescaled_bboxes = [bbox * scale for scale, bbox in zip(scales, bboxes)]
        # Change bboxes to RoI and RoI indices format
        mask_rois_before_reordering, mask_roi_indices_before_reordering =\
            _list_to_flat(rescaled_bboxes)
        mask_rois, mask_roi_indices, order = self.mask_head.distribute(
            mask_rois_before_reordering, mask_roi_indices_before_reordering)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            segms = F.sigmoid(
                self.mask_head(hs, mask_rois, mask_roi_indices)).data
        # Put the order of proposals back to the one used by bbox head
        # from the ordering respective FPN levels.
        segms = segms[order]
        segms = _flat_to_list(segms, mask_roi_indices_before_reordering)
        if len(segms) == 0:
            segms = [
                self.xp.zeros((0, self.mask_head.mask_size,
                               self.mask_head.mask_size), dtype=np.float32)]

        masks = self.mask_head.decode(
            segms,
            [bbox / scale for bbox, scale in zip(rescaled_bboxes, scales)],
            labels, sizes)

        masks = [cuda.to_cpu(mask) for mask in masks]
        labels = [cuda.to_cpu(label) for label in labels]
        scores = [cuda.to_cpu(score) for score in scores]
        return masks, labels, scores

    def prepare(self, imgs, masks=None):
        scales = []
        resized_imgs = []
        sizes = []
        for img in imgs:
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            scales.append(scale)
            H, W = int(H * scale), int(W * scale)
            img = transforms.resize(img, (H, W))
            img -= self.extractor.mean
            resized_imgs.append(img)
            sizes.append((H, W))
        pad_size = np.array(
            [im.shape[1:] for im in resized_imgs]).max(axis=0)
        pad_size = (
            np.ceil(pad_size / self._stride) * self._stride).astype(int)
        pad_imgs = np.zeros(
            (len(imgs), 3, pad_size[0], pad_size[1]), dtype=np.float32)
        for i, im in enumerate(resized_imgs):
            _, H, W = img.shape
            pad_imgs[i, :, :H, :W] = im
        pad_imgs = self.xp.array(pad_imgs)

        if masks is None:
            return pad_imgs, scales

        resized_masks = []
        for size, mask in zip(sizes, masks):
            resized_masks.append(transforms.resize(
                mask.astype(np.float32),
                size, interpolation=PIL.Image.NEAREST).astype(np.bool))
        pad_masks = []
        for mask in resized_masks:
            n_class, H, W = mask.shape
            pad_mask = self.xp.zeros(
                (n_class, pad_size[0], pad_size[1]), dtype=np.bool)
            pad_mask[:, :H, :W] = self.xp.array(mask)
            pad_masks.append(pad_mask)
        return pad_imgs, pad_masks, scales


def _list_to_flat(array_list):
    xp = chainer.backends.cuda.get_array_module(array_list[0])

    indices = xp.concatenate(
        [i * xp.ones((len(array),), dtype=np.int32) for
         i, array in enumerate(array_list)], axis=0)
    flat = xp.concatenate(array_list, axis=0)
    return flat, indices


def _flat_to_list(flat, indices):
    array_list = []
    for i in np.unique(chainer.backends.cuda.to_cpu(indices)):
        array_list.append(flat[indices == i])
    return array_list
