from __future__ import division

import itertools
import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv import transforms
from chainercv import utils


class SSD(chainer.Chain):
    """Base class of Single Shot Multibox Detector [1].

    This is a base class of Single Shot Multibox Detector.
    It requires a feature extraction method and a preprocessing method.

    [1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
    SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        extractor: A link which extract feature maps.
            This link must have :obj:`grid`, :meth:`prepare` and
            :meth:`__call__`.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        steps (iterable of float): The step size for each feature map.
        sizes (iterable of float): The base size of default bounding boxes
            for each feature map.
        variance (tuple of float): Two coefficients for encoding
            the locations of bounding boxe. The first value is used to
            encode coordinates of the centers. The second value is used to
            encode the sizes of bounding boxes.
            The default value is :obj:`(0.1, 0.2)`.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.GlorotUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    Parameters:
        nms_threshold (float): The threshold value
            for :meth:`chainercv.transfroms.non_maximum_suppression`.
            The default value is 0.45.
        score_threshold (float): The threshold value for confidence score.
            If a bounding box whose confidence score is lower than this value,
            the bounding box will be suppressed. The default value is 0.6.
            This value is optimized for visualization.
            For evaluation, the optimized value is 0.01.
    """

    def __init__(
            self, n_fg_class,
            extractor,
            aspect_ratios, steps, sizes,
            variance=(0.1, 0.2),
            initialW=initializers.GlorotUniform(),
            initial_bias=initializers.Zero()):
        self.n_fg_class = n_fg_class

        super(SSD, self).__init__(
            extractor=extractor,
            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )
        init = {'initialW': initialW, 'initial_bias': initial_bias}
        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(None, n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                None, n * (self.n_fg_class + 1), 3, pad=1, **init))

        # the format of default_bbox is (center_x, center_y, width, height)
        self._default_bbox = list()
        for k, grid in enumerate(extractor.grids):
            for v, u in itertools.product(range(grid), repeat=2):
                cx = (u + 0.5) * steps[k]
                cy = (v + 0.5) * steps[k]

                s = sizes[k]
                self._default_bbox.append((cx, cy, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                self._default_bbox.append((cx, cy, s, s))

                s = sizes[k]
                for ar in aspect_ratios[k]:
                    self._default_bbox.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    self._default_bbox.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        self._default_bbox = np.stack(self._default_bbox)

        self.variance = variance

        self.nms_threshold = 0.45
        self.score_threshold = 0.6

    def to_cpu(self):
        super(SSD, self).to_cpu()
        self._default_bbox = chainer.cuda.to_cpu(self._default_bbox)

    def to_gpu(self):
        super(SSD, self).to_gpu()
        self._default_bbox = chainer.cuda.to_gpu(self._default_bbox)

    def _multibox(self, xs):
        ys_loc = list()
        ys_conf = list()
        for i, x in enumerate(xs):
            loc = self.loc[i](x)
            loc = F.transpose(loc, (0, 2, 3, 1))
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            ys_loc.append(loc)

            conf = self.conf[i](x)
            conf = F.transpose(conf, (0, 2, 3, 1))
            conf = F.reshape(
                conf, (conf.shape[0], -1, self.n_fg_class + 1))
            ys_conf.append(conf)

        y_loc = F.concat(ys_loc, axis=1)
        y_conf = F.concat(ys_conf, axis=1)

        return y_loc, y_conf

    def __call__(self, x):
        """Compute localization and classification from a batch of images.

        This method computes two variables, :obj:`loc` and :obj:`conf`.
        :meth:`_decode` converts these variables to prediction.
        These variables are also used in training SSD.

        Args:
            x (chainer.Variable): A variable holding a batch of images.
                The images are preprocessed by :meth:`prepare` if needed.

        Returns:
            tuple of chainer.Variable:
            This method returns two variables, :obj:`loc` and :obj:`conf`.

            * **loc**: A variable of float arrays of shape :math:`(K, 4)`, \
                where :math:`K` is the number of default bounding boxes.
            * **conf**: A variable of float arrays of shape \
                :math:`(K, n\_class + 1)`, \
                where :math:`K` is the number of default bounding boxes.
        """

        return self._multibox(self.extractor(x))

    def _decode(self, loc, conf):
        xp = self.xp
        # the format of bbox is (center_x, center_y, width, height)
        bboxes = xp.dstack((
            self._default_bbox[:, :2] +
            loc[:, :, :2] * self.variance[0] * self._default_bbox[:, 2:],
            self._default_bbox[:, 2:] *
            xp.exp(loc[:, :, 2:] * self.variance[1])))
        # convert the format of bbox to (x_min, y_min, x_max, y_max)
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
        for l in range(1, 1 + self.n_fg_class):
            bbox_l = raw_bbox
            score_l = raw_score[:, l]

            mask = score_l >= self.score_threshold
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            if self.nms_threshold is not None:
                indices = utils.non_maximum_suppression(
                    bbox_l, self.nms_threshold, score_l)
                bbox_l = bbox_l[indices]
                score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append((l,) * len(bbox_l))
            score.append(score_l)

        bbox = xp.vstack(bbox)
        label = xp.hstack(label).astype(int)
        score = xp.hstack(score)

        return bbox, label, score

    def predict(self, imgs):
        """Detect objects from images

        This method predicts objects for each image.

        Args:
            imgs (iterable of ~numpy.ndarray): Arrays holding images.
                All images are in CHW and BGR format
                and the range of their value is :math:`[0, 255]`.

        Returns:
            tuple of list:
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
        sizes = list()
        for img in imgs:
            _, H, W = img.shape
            img = self.extractor.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            sizes.append((W, H))

        prepared_imgs = self.xp.stack(prepared_imgs)
        loc, conf = self(prepared_imgs)
        raw_bboxes, raw_scores = self._decode(loc.data, conf.data)

        bboxes = list()
        labels = list()
        scores = list()
        for raw_bbox, raw_score, size in zip(raw_bboxes, raw_scores, sizes):
            raw_bbox = transforms.resize_bbox(raw_bbox, (1, 1), size)
            bbox, label, score = self._suppress(raw_bbox, raw_score)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores
