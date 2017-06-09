from __future__ import division

import itertools
import numpy as np

import chainer

from chainercv import transforms
from chainercv import utils


class SSD(chainer.Chain):
    """Base class of Single Shot Multibox Detector.

    This is a base class of Single Shot Multibox Detector [#]_.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        extractor: A link which extracts feature maps.
            This link must have :obj:`insize`, :obj:`grids` and
            :meth:`__call__`.

            * :obj:`insize`: An integer which indicates \
            the size of input images. Images are resized to this size before \
            feature extraction.
            * :obj:`grids`: An iterable of integer. Each integer indicates \
            the size of feature map.
            * :meth:`__call_`: A method which computes feature maps. \
            It must take a batched images and return batched feature maps.
        multibox: A link which computes loc and conf from feature maps.
            This link must have :obj:`n_class`, :obj:`aspect_ratios` and
            :meth:`__call__`.

            * :obj:`n_class`: An integer which indicates the number of \
            classes. \
            This value should include the background class.
            * :obj:`aspect_ratios`: An iterable of tuple of integer. \
            Each tuple indicates the aspect ratios of default bounding boxes \
            at each feature maps.
            * :meth:`__call__`: A method which computes \
            :obj:`loc` and :obj:`conf`. \
            It must take a batched feature maps and \
            return :obj:`loc` and :obj:`conf`.
        steps (iterable of float): The step size for each feature map.
        sizes (iterable of float): The base size of default bounding boxes
            for each feature map.
        variance (tuple of float): Two coefficients for encoding
            the locations of bounding boxe. The first value is used to
            encode coordinates of the centers. The second value is used to
            encode the sizes of bounding boxes.
            The default value is :obj:`(0.1, 0.2)`.

    Parameters:
        nms_thresh (float): The threshold value
            for :meth:`chainercv.transfroms.non_maximum_suppression`.
            The default value is :obj:`0.45`.
            This value can be changed directly or by using :meth:`use_preset`.
        score_thresh (float): The threshold value for confidence score.
            If a bounding box whose confidence score is lower than this value,
            the bounding box will be suppressed.
            The default value is :obj:`0.6`.
            This value can be changed directly or by using :meth:`use_preset`.

    """

    def __init__(
            self, extractor, multibox,
            steps, sizes, variance=(0.1, 0.2),
            mean=0):
        self.variance = variance
        self.mean = mean
        self.use_preset('visualize')

        super(SSD, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.multibox = multibox

        # the format of default_bbox is (center_y, center_x, height, width)
        self._default_bbox = list()
        for k, grid in enumerate(extractor.grids):
            for v, u in itertools.product(range(grid), repeat=2):
                cy = (v + 0.5) * steps[k]
                cx = (u + 0.5) * steps[k]

                s = sizes[k]
                self._default_bbox.append((cy, cx, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                self._default_bbox.append((cy, cx, s, s))

                s = sizes[k]
                for ar in multibox.aspect_ratios[k]:
                    self._default_bbox.append(
                        (cy, cx, s / np.sqrt(ar), s * np.sqrt(ar)))
                    self._default_bbox.append(
                        (cy, cx, s * np.sqrt(ar), s / np.sqrt(ar)))
        self._default_bbox = np.stack(self._default_bbox)

    @property
    def insize(self):
        return self.extractor.insize

    @property
    def n_fg_class(self):
        return self.multibox.n_class - 1

    def to_cpu(self):
        super(SSD, self).to_cpu()
        self._default_bbox = chainer.cuda.to_cpu(self._default_bbox)

    def to_gpu(self, device=None):
        super(SSD, self).to_gpu(device)
        self._default_bbox = chainer.cuda.to_gpu(
            self._default_bbox, device=device)

    def __call__(self, x):
        """Compute localization and classification from a batch of images.

        This method computes two variables, :obj:`loc` and :obj:`conf`.
        :meth:`_decode` converts these variables to bounding box coordinates
        and confidence scores.
        These variables are also used in training SSD.

        Args:
            x (chainer.Variable): A variable holding a batch of images.
                The images are preprocessed by :meth:`_prepare`.

        Returns:
            tuple of chainer.Variable:
            This method returns two variables, :obj:`loc` and :obj:`conf`.

            * **loc**: A variable of float arrays of shape :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                ::math:`K` is the number of default bounding boxes.
            * **conf**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.
        """

        return self.multibox(self.extractor(x))

    def _decode(self, loc, conf):
        xp = self.xp
        # the format of bbox is (center_y, center_x, height, width)
        bboxes = xp.dstack((
            self._default_bbox[:, :2] +
            loc[:, :, :2] * self.variance[0] * self._default_bbox[:, 2:],
            self._default_bbox[:, 2:] *
            xp.exp(loc[:, :, 2:] * self.variance[1])))
        # convert the format of bbox to (y_min, x_min, y_max, x_max)
        bboxes[:, :, :2] -= bboxes[:, :, 2:] / 2
        bboxes[:, :, 2:] += bboxes[:, :, :2]
        scores = xp.exp(conf)
        scores /= scores.sum(axis=2, keepdims=True)
        return bboxes, scores

    def _suppress(self, raw_bbox, raw_score):
        xp = self.xp

        bbox = list()
        label = list()
        score = list()
        for l in range(self.n_fg_class):
            bbox_l = raw_bbox
            # the l-th class corresponds for the (l + 1)-th column.
            score_l = raw_score[:, l + 1]

            mask = score_l >= self.score_thresh
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            if self.nms_thresh is not None:
                indices = utils.non_maximum_suppression(
                    bbox_l, self.nms_thresh, score_l)
                bbox_l = bbox_l[indices]
                score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append(xp.array((l,) * len(bbox_l)))
            score.append(score_l)

        bbox = xp.vstack(bbox)
        label = xp.hstack(label).astype(int)
        score = xp.hstack(score)

        return bbox, label, score

    def _prepare(self, img):
        img = img.astype(np.float32)
        img = transforms.resize(img, (self.insize, self.insize))
        img -= np.array(self.mean)[:, np.newaxis, np.newaxis]
        return img

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
            self.nms_thresh = 0.45
            self.score_thresh = 0.6
        elif preset == 'evaluate':
            self.nms_thresh = 0.45
            self.score_thresh = 0.01
        else:
            raise ValueError('preset must be visualize or evaluate')

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
               Each bouding box is organized by \
               :obj:`(y_min, x_min, y_max, x_max)` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """

        x = list()
        sizes = list()
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))

        with chainer.function.no_backprop_mode():
            x = chainer.Variable(self.xp.stack(x))
            loc, conf = self(x)
        raw_bboxes, raw_scores = self._decode(loc.data, conf.data)

        bboxes = list()
        labels = list()
        scores = list()
        for raw_bbox, raw_score, size in zip(raw_bboxes, raw_scores, sizes):
            raw_bbox = transforms.resize_bbox(raw_bbox, (1, 1), size)
            bbox, label, score = self._suppress(raw_bbox, raw_score)
            bboxes.append(chainer.cuda.to_cpu(bbox))
            labels.append(chainer.cuda.to_cpu(label))
            scores.append(chainer.cuda.to_cpu(score))

        return bboxes, labels, scores
