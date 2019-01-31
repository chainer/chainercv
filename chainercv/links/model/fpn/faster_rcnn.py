from __future__ import division

import numpy as np

import chainer
from chainer.backends import cuda

from chainercv import transforms


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
        head (Link): A link that has the same interface as
            :class:`~chainercv.links.model.fpn.Head`.
            Please refer to the documentation found there.

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

    _min_size = 800
    _max_size = 1333
    _stride = 32

    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

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
        head_locs, head_confs = self.head(hs, rois, roi_indices)
        return rois, roi_indices, head_locs, head_confs

    def predict(self, imgs):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bounding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """

        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            rois, roi_indices, head_locs, head_confs = self(x)
        bboxes, labels, scores = self.head.decode(
            rois, roi_indices, head_locs, head_confs,
            scales, sizes, self.nms_thresh, self.score_thresh)

        bboxes = [cuda.to_cpu(bbox) for bbox in bboxes]
        labels = [cuda.to_cpu(label) for label in labels]
        scores = [cuda.to_cpu(score) for score in scores]
        return bboxes, labels, scores

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
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            scales.append(scale)
            H, W = int(H * scale), int(W * scale)
            img = transforms.resize(img, (H, W))
            img -= self.extractor.mean
            resized_imgs.append(img)

        size = np.array([im.shape[1:] for im in resized_imgs]).max(axis=0)
        size = (np.ceil(size / self._stride) * self._stride).astype(int)
        x = np.zeros((len(imgs), 3, size[0], size[1]), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img

        x = self.xp.array(x)
        return x, scales
