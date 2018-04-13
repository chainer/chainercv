from __future__ import division

import chainer
import chainer.functions as F
import numpy as np

from chainercv.experimental.links.model.fcis.utils.mask_voting \
    import mask_voting
from chainercv.transforms.image.resize import resize


class FCIS(chainer.Chain):

    def __init__(
            self, extractor, rpn, head,
            mean, min_size, max_size,
            loc_normalize_mean, loc_normalize_std,
    ):
        super(FCIS, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.use_preset('visualize')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]

        # Feature Extractor
        rpn_features, roi_features = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            rpn_features, img_size, scale)
        roi_seg_scores, roi_ag_locs, roi_scores, rois, roi_indices = \
            self.head(roi_features, rois, roi_indices, img_size)
        return roi_seg_scores, roi_ag_locs, roi_scores, rois, roi_indices

    def prepare(self, img):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        _, H, W = img.shape

        scale = self.min_size / min(H, W)

        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)

        img = resize(img, (int(H * scale), int(W * scale)))
        img = (img - self.mean).astype(np.float32, copy=False)
        return img

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
            self.mask_merge_thresh = 0.5
            self.binary_thresh = 0.4
            self.limit = 100
            self.min_drop_size = 16
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 1e-3
            self.mask_merge_thresh = 0.5
            self.binary_thresh = 0.4
            self.limit = 100
            self.min_drop_size = 16
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs):

        prepared_imgs = []
        sizes = []
        for img in imgs:
            size = img.shape[1:]
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            sizes.append(size)

        masks = []
        labels = []
        scores = []
        bboxes = []

        for img, size in zip(prepared_imgs, sizes):
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                # inference
                img_var = chainer.Variable(self.xp.array(img[None]))
                scale = img_var.shape[3] / size[1]
                roi_seg_scores, roi_ag_locs, roi_scores, rois, _ = \
                    self.__call__(img_var, scale)

            roi_seg_score = roi_seg_scores.array
            roi_score = roi_scores.array
            bbox = rois / scale

            # shape: (n_rois, 4)
            bbox[:, 0::2] = self.xp.clip(bbox[:, 0::2], 0, size[0])
            bbox[:, 1::2] = self.xp.clip(bbox[:, 1::2], 0, size[1])

            roi_seg_prob = F.softmax(roi_seg_score).array
            roi_prob = F.softmax(roi_score).array

            roi_seg_prob = chainer.cuda.to_cpu(roi_seg_prob)
            roi_prob = chainer.cuda.to_cpu(roi_prob)
            bbox = chainer.cuda.to_cpu(bbox)

            roi_mask_prob, label, score, bbox = mask_voting(
                roi_seg_prob[:, 1, :, :], roi_prob, bbox,
                size, self.n_class,
                self.score_thresh, self.nms_thresh,
                self.mask_merge_thresh, self.binary_thresh,
                limit=self.limit, bg_label=0)

            height = bbox[:, 2] - bbox[:, 0]
            width = bbox[:, 3] - bbox[:, 1]
            keep_indices = np.where(
                (height > self.min_drop_size) &
                (width > self.min_drop_size))[0]
            bbox = bbox[keep_indices]
            roi_mask_prob = roi_mask_prob[keep_indices]
            score = score[keep_indices]
            label = label[keep_indices]

            mask = np.zeros(
                (len(roi_mask_prob), size[0], size[1]), dtype=np.bool)
            for i, (roi_msk_prb, bb) in enumerate(zip(roi_mask_prob, bbox)):
                bb = np.round(bb).astype(np.int32)
                y_min, x_min, y_max, x_max = bb
                roi_msk_prb = resize(
                    roi_msk_prb.astype(np.float32)[None],
                    (y_max - y_min, x_max - x_min))
                roi_msk = (roi_msk_prb > self.binary_thresh)[0]
                mask[i, y_min:y_max, x_min:x_max] = roi_msk

            masks.append(mask)
            labels.append(label)
            scores.append(score)
            bboxes.append(bbox)

        return masks, labels, scores, bboxes
