from __future__ import division

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F

from chainercv.links.model.mask_rcnn.misc import scale_img


class MaskRCNN(chainer.Chain):

    """Base class of Mask R-CNN.

    This is a base class of Mask R-CNN [#]_.

    .. [#] Kaiming He et al. Mask R-CNN. ICCV 2017

    Args:
        extractor (Link): A link that extracts feature maps.
            This link must have :obj:`scales`, :obj:`mean` and
            :meth:`__call__`.
        rpn (Link): A link that has the same interface as
            :class:`~chainercv.links.model.fpn.RPN`.
            Please refer to the documentation found there.
        head (Link): A link that has the same interface as
            :class:`~chainercv.links.model.fpn.Head`.
            Please refer to the documentation found there.
        mask_head (Link): A link that has the same interface as
            :class:`~chainercv.links.model.mask_rcnn.MaskRCNN`.
            Please refer to the documentation found there.

    Parameters:
        nms_thresh (float): The threshold value
            for :func:`~chainercv.utils.non_maximum_suppression`.
            The default value is :obj:`0.5`.
            This value can be changed directly or by using :meth:`use_preset`.
        score_thresh (float): The threshold value for confidence score.
            If a bounding box whose confidence score is lower than this value,
            the bounding box will be suppressed.
            The default value is :obj:`0.7`.
            This value can be changed directly or by using :meth:`use_preset`.

    """

    min_size = 800
    max_size = 1333
    stride = 32

    def __init__(self, extractor, rpn, head, mask_head):
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head
            self.mask_head = mask_head

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
        rois, roi_indices = self.head.distribute(rois, roi_indices)
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

        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            hs, rois, roi_indices = self(x)
            head_locs, head_confs = self.head(hs, rois, roi_indices)
        bboxes, labels, scores = self.head.decode(
            rois, roi_indices, head_locs, head_confs,
            scales, sizes, self.nms_thresh, self.score_thresh)

        rescaled_bboxes = [bbox * scale for scale, bbox in zip(scales, bboxes)]
        # Change bboxes to RoI and RoI indices format
        mask_rois_before_reordering, mask_roi_indices_before_reordering =\
            _list_to_flat(rescaled_bboxes)
        mask_rois, mask_roi_indices, order = self.mask_head.distribute(
            mask_rois_before_reordering, mask_roi_indices_before_reordering)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            segms = F.sigmoid(
                self.mask_head(hs, mask_rois, mask_roi_indices)).data
        # Put the order of proposals back to the one used by bbox head.
        segms = segms[order]
        segms = _flat_to_list(
            segms, mask_roi_indices_before_reordering, len(imgs))
        segms = [segm if segm is not None else
                 self.xp.zeros(
                     (0, self.mask_head.mask_size, self.mask_head.mask_size),
                     dtype=np.float32)
                 for segm in segms]

        segms = [chainer.backends.cuda.to_cpu(segm) for segm in segms]
        bboxes = [chainer.backends.cuda.to_cpu(bbox / scale)
                  for bbox, scale in zip(rescaled_bboxes, scales)]
        labels = [chainer.backends.cuda.to_cpu(label) for label in labels]
        # Currently MaskHead only supports numpy inputs
        masks = self.mask_head.decode(segms, bboxes, labels, sizes)
        scores = [cuda.to_cpu(score) for score in scores]
        return masks, labels, scores

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
