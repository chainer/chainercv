# Mofidied work:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# --------------------------------------------------------
#
# Original works by:
# --------------------------------------------------------
# Faster R-CNN implementation by Chainer
# Copyright (c) 2016 Shunta Saito
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/mitmul/chainer-faster-rcnn
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

from __future__ import division

import copy
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression

from chainercv.transforms.image.resize import resize


class FasterRCNNBase(chainer.Chain):

    """Base class for Faster RCNN.

    This is a base class for Faster RCNN [1].
    The following three stages constitute Faster RCNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using features that belong \
        to the proposed RoIs, classify the categories of the objects in the \
        RoIs and improve localizations.

    Each stage is carried out by one of the callable objects: :obj:`feature`,
    :obj:`rpn` and :obj:`head`.

    There are two functions :func:`predict` and :func:`__call__` to conduct
    object detection.
    :func:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster RCNN is treated as a black box function, for instance.
    :func:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance in training and debugging.

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        feature (callable): A callable that takes BCHW image array and option
            :obj:`train` as arguments, and returns a BCHW feature.
        rpn (callable): A callable that has same interface as
            :class:`chainercv.links.RegionProposalNetwork`. Please refer to
            the documentation found there.
        head (callable): A callable that takes tuple of BCHW array,
            RoIs and batch indices for RoIs. This returns class dependent
            localization paramters and class scores.
        n_class (int): The number of classes including the background.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :func:`prepare`.
        nms_thresh (float): Threshold value used when calling non maximum
            suppression in :func:`predict`.
        score_thresh (float): Threshold value used to discard low
            confidence proposals in :func:`predict`.
        min_size (int): A preprocessing paramter for :func:`prepare`.
        max_size (int): A preprocessing paramter for :func:`prepare`.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(
            self, feature, rpn, head,
            n_class, mean,
            nms_thresh=0.3,
            score_thresh=0.7,
            min_size=600,
            max_size=1000,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
    ):
        super(FasterRCNNBase, self).__init__(
            feature=feature,
            rpn=rpn,
            head=head,
        )
        self.n_class = n_class
        self.mean = mean
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def _decide_when_to_stop(self, layers):
        layers = copy.copy(layers)
        if len(layers) == 0:
            return 'start'

        rpn_outs = [
            'features', 'rpn_locs', 'rpn_scores',
            'rois', 'batch_indices', 'anchor']
        for layer in rpn_outs:
            layers.pop(layer, None)

        if len(layers) == 0:
            return 'rpn'
        return 'head'

    def _update_if_specified(self, target, source):
        for key in source.keys():
            if key in target:
                target[key] = source[key]

    def __call__(self, x, scale=1.,
                 layers=['rois', 'roi_cls_locs', 'roi_scores'], test=True):
        """Computes all the values specified by :obj:`layers`.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`H, W` are the height and the width of the features extracted \
            by :obj:`feature`.
        * :math:`R'` is the total number of RoIs produced across batches.
        * :math:`C` is the number of feature channels.
        * :math:`L` is the number of classes
        * :math:`A` is the number of anchors per pixel.

        Here are list of the names, types and descriptions of values that can
        be collected.

        * **features** (*Variable*): Feature extractor output \
            (e.g. conv5\_3 for VGG16). Its shape is :math:`(N, C, H, W)`.
        * **rpn_locs** (*Variable*): Bounding box offsets for each anchor. \
            Its shape is :math:`(N, HWA, 4)`.
        * **rpn_scores** (*Variable*): Confidence scores for each anchor to \
            be a foreground anchor. Its shape is :math:`(N, HWA, 2)`.
        * **rois** (*array*): RoIs produced by RPN. Its shape is \
            :math:`(R', 4)`.
        * **batch_indices** (*array*): Batch indices of RoIs. Its shape is \
            :math:`(R',)`.
        * **anchor** (*array*): Anchors used by RPN. Its shape is \
            :math:`(HWA, 4)`.
        * **roi_cls_locs** (*Variable*): Bounding box offsets for RoIs. \
            Its shape is :math:`(R', L4)`.
        * **roi_scores** (*Variable*): Class predictions for RoIs. \
            Its shape is :math:`(R', L)`.

        If none of the features need to be collected after RPN,
        the function returns after finish calling RPN without using the head
        of the network.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Args:
            x (~chainer.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                in preprocessing.
            layers (list of str): The list of the names of the values to be
                collected.
            test (bool): If :obj:`True`, test time behavior is used.

        Returns:
            Dictionary of variables and arrays:
            A directory whose key corresponds to the layer name specified by \
            :obj:`layers`.

        """
        activations = {key: None for key in layers}
        stop_at = self._decide_when_to_stop(activations)
        if stop_at == 'start':
            return {}

        img_size = x.shape[2:][::-1]

        h = self.feature(x, train=not test)
        rpn_locs, rpn_scores, rois, batch_indices, anchor =\
            self.rpn(h, img_size, scale, train=not test)

        self._update_if_specified(
            activations,
            {'features': h,
             'rpn_locs': rpn_locs,
             'rpn_scores': rpn_scores,
             'rois': rois,
             'batch_indices': batch_indices,
             'anchor': anchor})
        if stop_at == 'rpn':
            return activations

        roi_cls_locs, roi_scores = self.head(
            h, rois, batch_indices, train=not test)
        self._update_if_specified(
            activations,
            {'roi_cls_locs': roi_cls_locs,
             'roi_scores': roi_scores})
        return activations

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox[:, l * 4: (l + 1) * 4]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            label.append(l * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def predict(self, imgs):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and BGR format
                and the range of their values are :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :obj:`(x_min, y_min, x_max, y_max)` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        prepared_imgs = list()
        scales = list()
        for img in imgs:
            _, H, W = img.shape
            img = self.prepare(img.astype(np.float32))
            scale = img.shape[2] / W
            prepared_imgs.append(img)
            scales.append(scale)

        bboxes = list()
        labels = list()
        scores = list()
        for img, scale in zip(prepared_imgs, scales):
            img_var = chainer.Variable(
                self.xp.asarray(img[None]), volatile=chainer.flag.ON)
            H, W = img_var.shape[2:]
            out = self.__call__(
                img_var, scale=scale,
                layers=['rois', 'roi_cls_locs', 'roi_scores'],
                test=True)
            # We are assuming that batch size is 1.
            roi_cls_loc = out['roi_cls_locs'].data
            roi_score = out['roi_scores'].data
            roi = out['rois'] / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean),
                                self.n_class)
            std = self.xp.tile(self.xp.asarray(self.loc_normalize_std),
                               self.n_class)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
            roi_cls_loc = roi_cls_loc.reshape(-1, self.n_class, 4)
            roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)
            cls_bbox = loc2bbox(roi.reshape(-1, 4), roi_cls_loc.reshape(-1, 4))
            cls_bbox = cls_bbox.reshape(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, slice(0, 4, 2)] = self.xp.clip(
                cls_bbox[:, slice(0, 4, 2)], 0, W / scale)
            cls_bbox[:, slice(1, 4, 2)] = self.xp.clip(
                cls_bbox[:, slice(1, 4, 2)], 0, H / scale)

            prob = F.softmax(roi_score).data

            raw_cls_bbox = cuda.to_cpu(cls_bbox)
            raw_prob = cuda.to_cpu(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores

    def prepare(self, img):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After that, if the length of the longer edge is longer than
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing, image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and BGR format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        _, H, W = img.shape

        scale = 1.
        if min(H, W) < self.min_size:
            scale = self.min_size / min(H, W)

        if scale * max(H, W) > self.max_size:
            scale = max(H, W) * scale / self.max_size

        img = resize(img, (int(W * scale), int(H * scale)))

        img = (img - self.mean).astype(np.float32, copy=False)
        return img
