from __future__ import division

import chainer
import chainer.functions as F
import numpy as np

from chainercv.experimental.links.model.fcis.utils.mask_voting \
    import mask_voting
from chainercv.transforms.image.resize import resize


class FCIS(chainer.Chain):

    """Base class for FCIS.

    This is a base class for FCIS links supporting instance segmentation
    API [#]_. The following three stages constitute FCIS.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization, Segmentation and Classification Heads**: Using feature \
        maps that belong to the proposed RoIs, segment regions of the \
        objects, classify the categories of the objects in the RoIs and \
        improve localizations.

    Each stage is carried out by one of the callable
    :class:`chainer.Chain` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.
    There are two functions :meth:`predict` and :meth:`forward` to conduct
    instance segmentation.
    :meth:`predict` takes images and returns masks, object labels
    and their scores.
    :meth:`forward` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support instance segmentation API have method :meth:`predict`
    with the same interface. Please refer to :meth:`predict` for further
    details.

    .. [#] Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei. \
    Fully Convolutional Instance-aware Semantic Segmentation. CVPR 2017.

    Args:
        extractor (callable Chain): A callable that takes a BCHW image
            array and returns feature maps.
        rpn (callable Chain): A callable that has the same interface as
            :class:`~chainercv.links.model.faster_rcnn.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (callable Chain): A callable that takes a BCHW array,
            RoIs and batch indices for RoIs.
            This returns class-agnostic segmentation scores, class-agnostic
            localization parameters, class scores, improved RoIs and batch
            indices for RoIs.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :meth:`prepare`.
        min_size (int): A preprocessing parameter for :meth:`prepare`. Please
            refer to a docstring found for :meth:`prepare`.
        max_size (int): A preprocessing parameter for :meth:`prepare`.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

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

    def forward(self, x, scales=None):
        """Forward FCIS.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.
        * :math:`RH` is the height of pooled image by Position Sensitive \
            ROI pooling.
        * :math:`RW` is the height of pooled image by Position Sensitive \
            ROI pooling.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (~chainer.Variable): 4D image variable.
            scales (tuple of floats): Amount of scaling applied to each input
                image during preprocessing.

        Returns:
            Variable, Variable, Variable, array, array:
            Returns tuple of five values listed below.

            * **roi_ag_seg_scores**: Class-agnostic clipped mask scores for \
                the proposed ROIs. Its shape is :math:`(R', 2, RH, RW)`
            * **ag_locs**: Class-agnostic offsets and scalings for \
                the proposed RoIs.  Its shape is :math:`(R', 2, 4)`.
            * **roi_cls_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]

        # Feature Extractor
        rpn_features, roi_features = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            rpn_features, img_size, scales)
        roi_ag_seg_scores, roi_ag_locs, roi_cls_scores, rois, roi_indices = \
            self.head(roi_features, rois, roi_indices, img_size)
        return roi_ag_seg_scores, roi_ag_locs, roi_cls_scores, \
            rois, roi_indices

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

        This method changes values of :obj:`self.nms_thresh`,
        :obj:`self.score_thresh`, :obj:`self.mask_merge_thresh`,
        :obj:`self.binary_thresh`, :obj:`self.binary_thresh` and
        :obj:`self.min_drop_size`. These values are a threshold value
        used for non maximum suppression, a threshold value
        to discard low confidence proposals in :meth:`predict`,
        a threshold value to merge mask in :meth:`predict`,
        a threshold value to binalize segmentation scores in :meth:`predict`,
        a limit number of predicted masks in one image and
        a threshold value to discard small bounding boxes respectively.

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
        elif preset == 'coco_evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 1e-3
            self.mask_merge_thresh = 0.5
            self.binary_thresh = 0.4
            self.limit = 100
            self.min_drop_size = 2
        else:
            raise ValueError('preset must be visualize or evaluate')

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

        for img, size in zip(prepared_imgs, sizes):
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                # inference
                img_var = chainer.Variable(self.xp.array(img[None]))
                scale = img_var.shape[3] / size[1]
                roi_ag_seg_scores, _, roi_cls_scores, bboxes, _ = \
                    self.forward(img_var, scales=[scale])

            # We are assuming that batch size is 1.
            roi_ag_seg_score = chainer.cuda.to_cpu(roi_ag_seg_scores.array)
            roi_cls_score = chainer.cuda.to_cpu(roi_cls_scores.array)
            bbox = chainer.cuda.to_cpu(bboxes)

            # filter bounding boxes with min_size
            height = bbox[:, 2] - bbox[:, 0]
            width = bbox[:, 3] - bbox[:, 1]
            keep_indices = np.where(
                (height >= self.min_drop_size) &
                (width >= self.min_drop_size))[0]
            roi_ag_seg_score = roi_ag_seg_score[keep_indices, :, :]
            roi_cls_score = roi_cls_score[keep_indices]
            bbox = bbox[keep_indices, :]

            # scale bbox
            bbox = bbox / scale

            # shape: (n_rois, 4)
            bbox[:, 0::2] = self.xp.clip(bbox[:, 0::2], 0, size[0])
            bbox[:, 1::2] = self.xp.clip(bbox[:, 1::2], 0, size[1])

            # shape: (n_roi, roi_size, roi_size)
            roi_seg_prob = F.softmax(roi_ag_seg_score).array[:, 1]
            roi_cls_prob = F.softmax(roi_cls_score).array

            roi_seg_prob, bbox, label, roi_cls_prob = mask_voting(
                roi_seg_prob, bbox, roi_cls_prob, size,
                self.score_thresh, self.nms_thresh,
                self.mask_merge_thresh, self.binary_thresh,
                limit=self.limit, bg_label=0)

            mask = np.zeros(
                (len(roi_seg_prob), size[0], size[1]), dtype=np.bool)
            for i, (roi_seg_pb, bb) in enumerate(zip(roi_seg_prob, bbox)):
                bb = np.round(bb).astype(np.int32)
                y_min, x_min, y_max, x_max = bb
                roi_msk_pb = resize(
                    roi_seg_pb.astype(np.float32)[None],
                    (y_max - y_min, x_max - x_min))
                roi_msk = (roi_msk_pb > self.binary_thresh)[0]
                mask[i, y_min:y_max, x_min:x_max] = roi_msk

            masks.append(mask)
            labels.append(label)
            scores.append(roi_cls_prob)

        return masks, labels, scores
