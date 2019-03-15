from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
from chainer.backends import cuda

from chainercv.links.model.fpn.misc import scale_img


class FasterRCNN(chainer.Chain):
    """Base class of Feature Pyramid Networks.

    This is a base class of Feature Pyramid Networks [#]_.

    .. [#] Tsung-Yi Lin et al.
       Feature Pyramid Networks for Object Detection. CVPR 2017

    Args:
        extractor (Link): A link that extracts feature maps.
            This link must have :obj:`scales`, :obj:`mean` and
            :meth:`__call__`.
        rpn (Link): A link that has the same interface as
            :class:`~chainercv.links.model.fpn.RPN`.
            Please refer to the documentation found there.
        bbox_head (Link): A link that has the same interface as
            :class:`~chainercv.links.model.fpn.BboxHead`.
            Please refer to the documentation found there.
        mask_head (Link): A link that has the same interface as
            :class:`~chainercv.links.model.mask_rcnn.MaskRCNN`.
            Please refer to the documentation found there.
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please
            refer to a docstring found for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`. Note
            that the result of :meth:`prepare` can exceed this size due to
            alignment with stride.

    Parameters:
        nms_thresh (float): The threshold value
            for :func:`~chainercv.utils.non_maximum_suppression`.
            The default value is :obj:`0.45`.
            This value can be changed directly or by using :meth:`use_preset`.
        score_thresh (float): The threshold value for confidence score.
            If a bounding box whose confidence score is lower than this value,
            the bounding box will be suppressed.
            The default value is :obj:`0.6`.
            This value can be changed directly or by using :meth:`use_preset`.

    """

    stride = 32
    _accepted_return_values = ('rois', 'bboxes', 'labels', 'scores', 'masks')

    def __init__(self, extractor, rpn, bbox_head,
                 mask_head, return_values,
                 min_size=800, max_size=1333):
        for value_name in return_values:
            if value_name not in self._accepted_return_values:
                raise ValueError(
                    '{} is not included in accepted value names {}'.format(
                        value_name, self._accepted_return_values))
        self._return_values = return_values

        self._store_rpn_outputs = 'rois' in self._return_values
        self._run_bbox = any([key in self._return_values
                        for key in ['bboxes', 'labels', 'scores', 'masks']])
        self._run_mask = 'masks' in self._return_values
        super(FasterRCNN, self).__init__()

        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            if self._run_bbox:
                self.bbox_head = bbox_head
            if self._run_mask:
                self.mask_head = mask_head

        self.min_size = min_size
        self.max_size = max_size

        self.use_preset('visualize')

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`nms_thresh` and
        :obj:`score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'}): A string to determine the
                preset to use.
        """

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
        return hs, rois, roi_indices

    def predict(self, imgs):
        """Segment object instances from images.

        This method predicts instance-aware object regions for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images of shape
                :math:`(B, C, H, W)`.  All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(masks, labels, scores)`.

           * **masks**: A list of boolean arrays of shape :math:`(R, H, W)`, \
               where :math:`R` is the number of masks in a image. \
               Each pixel holds value if it is inside the object inside or not.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the masks. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        output = {}

        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            hs, rpn_rois, rpn_roi_indices = self(x)
            if self._store_rpn_outputs:
                rpn_rois_cpu = [
                    chainer.backends.cuda.to_cpu(rpn_roi) / scale
                    for rpn_roi, scale in
                    zip(_flat_to_list(rpn_rois, rpn_roi_indices, len(imgs)),
                        scales)]
                output.update({'rois': rpn_rois_cpu})

        if self._run_bbox:
            bbox_rois, bbox_roi_indices = self.bbox_head.distribute(
                rpn_rois, rpn_roi_indices)
            with chainer.using_config(
                    'train', False), chainer.no_backprop_mode():
                head_locs, head_confs = self.bbox_head(
                    hs, bbox_rois, bbox_roi_indices)
            bboxes, labels, scores = self.bbox_head.decode(
                bbox_rois, bbox_roi_indices, head_locs, head_confs,
                scales, sizes, self.nms_thresh, self.score_thresh)
            bboxes_cpu = [chainer.backends.cuda.to_cpu(bbox)
                    for bbox in bboxes]
            labels_cpu = [chainer.backends.cuda.to_cpu(label) for label in labels]
            scores_cpu = [cuda.to_cpu(score) for score in scores]
            output.update({'bboxes': bboxes_cpu, 'labels': labels_cpu,
                           'scores': scores_cpu})

        if self._run_mask:
            rescaled_bboxes = [bbox * scale
                               for scale, bbox in zip(scales, bboxes)]
            # Change bboxes to RoI and RoI indices format
            mask_rois_before_reordering, mask_roi_indices_before_reordering =\
                _list_to_flat(rescaled_bboxes)
            mask_rois, mask_roi_indices, order = self.mask_head.distribute(
                mask_rois_before_reordering, mask_roi_indices_before_reordering)
            with chainer.using_config(
                    'train', False), chainer.no_backprop_mode():
                segms = F.sigmoid(
                    self.mask_head(hs, mask_rois, mask_roi_indices)).data
            # Put the order of proposals back to the one used by bbox head.
            segms = segms[order]
            segms = _flat_to_list(
                segms, mask_roi_indices_before_reordering, len(imgs))
            segms = [segm if segm is not None else
                    self.xp.zeros(
                        (0, self.mask_head.segm_size, self.mask_head.segm_size),
                        dtype=np.float32)
                    for segm in segms]
            segms = [chainer.backends.cuda.to_cpu(segm) for segm in segms]
            # Currently MaskHead only supports numpy inputs
            masks_cpu = self.mask_head.decode(segms, bboxes_cpu, labels_cpu, sizes)
            output.update({'masks': masks_cpu})
        return tuple([output[key] for key in self._return_values])

    def prepare(self, imgs):
        """Preprocess images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
            Two arrays: preprocessed images and \
            scales that were caluclated in prepocessing.

        """
        scales = []
        resized_imgs = []
        for img in imgs:
            img, scale = scale_img(
                img, self.min_size, self.max_size)
            img -= self.extractor.mean
            scales.append(scale)
            resized_imgs.append(img)
        pad_size = np.array(
            [im.shape[1:] for im in resized_imgs]).max(axis=0)
        pad_size = (
            np.ceil(pad_size / self.stride) * self.stride).astype(int)
        x = np.zeros(
            (len(imgs), 3, pad_size[0], pad_size[1]), dtype=np.float32)
        for i, im in enumerate(resized_imgs):
            _, H, W = im.shape
            x[i, :, :H, :W] = im
        x = self.xp.array(x)

        return x, scales


def _list_to_flat(array_list):
    xp = chainer.backends.cuda.get_array_module(array_list[0])

    indices = xp.concatenate(
        [i * xp.ones((len(array),), dtype=np.int32) for
         i, array in enumerate(array_list)], axis=0)
    flat = xp.concatenate(array_list, axis=0)
    return flat, indices


def _flat_to_list(flat, indices, B):
    array_list = []
    for i in range(B):
        array = flat[indices == i]
        if len(array) > 0:
            array_list.append(array)
        else:
            array_list.append(None)
    return array_list
