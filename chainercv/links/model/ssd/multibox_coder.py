from __future__ import division

import itertools
import numpy as np

import chainer

from chainercv import utils


class MultiboxCoder(object):
    """A helper class to encode/decode bounding boxes.

    This class encodes :obj:`(bbox, label)` to :obj:`(mb_loc, mb_label)`
    and decodes :obj:`(mb_loc, mb_conf)` to `(bbox, label, score)`.
    These encoding/decoding are used in Single Shot Multibox Detector [#]_.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        grids (iterable of ints): An iterable of integers.
            Each integer indicates the size of feature map.
        aspect_ratios (iterable of tuples of ints)`:
            An iterable of tuples of integers.
            Each tuple indicates the aspect ratios of default bounding boxes
            at each feature maps.
            The length of this iterable should be :obj:`len(grids)`.
        steps (iterable of floats): The step size for each feature map.
            The length of this iterable should be :obj:`len(grids)`.
        sizes (iterable of floats): The base size of default bounding boxes
            for each feature map.
            The length of this iterable should be :obj:`len(grids) + 1`.
        variance (tuple of floats): Two coefficients for encoding/decoding
            the locations of bounding boxe. The first value is used to
            encode/decode coordinates of the centers.
            The second value is used to encode/decode the sizes of
            bounding boxes.
    """

    def __init__(self, grids, aspect_ratios, steps, sizes, variance):
        if not len(aspect_ratios) == len(grids):
            raise ValueError('The length of aspect_ratios is wrong.')
        if not len(steps) == len(grids):
            raise ValueError('The length of steps is wrong.')
        if not len(sizes) == len(grids) + 1:
            raise ValueError('The length of sizes is wrong.')

        default_bbox = list()

        for k, grid in enumerate(grids):
            for v, u in itertools.product(range(grid), repeat=2):
                cx = (u + 0.5) * steps[k]
                cy = (v + 0.5) * steps[k]

                s = sizes[k]
                default_bbox.append((cx, cy, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                default_bbox.append((cx, cy, s, s))

                s = sizes[k]
                for ar in aspect_ratios[k]:
                    default_bbox.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    default_bbox.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))

        # the format of _default_bbox is (center_x, center_y, width, height)
        self._default_bbox = np.stack(default_bbox)
        self._variance = variance

    @property
    def xp(self):
        return chainer.cuda.get_array_module(self._default_bbox)

    def to_cpu(self):
        self._default_bbox = chainer.cuda.to_cpu(self._default_bbox)

    def to_gpu(self, device=None):
        self._default_bbox = chainer.cuda.to_gpu(
            self._default_bbox, device=device)

    def encode(self, bbox, label, iou_thresh=0.5):
        """Encodes coordinates and classes of bounding boxes.

        This method encodes :obj:`bbox: and :obj:`label` to :obj:`mb_loc`
        and :obj:`mb_label`, which are used to compute multibox loss.

        Args:
            bbox (array): A float array of shape :math:`(R, 4)`,
                where :math:`R` is the number of bounding boxes in a image.
                Each bouding box is organized by
                :obj:`(x_min, y_min, x_max, y_max)`
                in the second axis.
            label (array) : An integer array of shape :math:`(R,)`.
                Each value indicates the class of the bounding box.
            iou_thresh (float): The threshold value to determine
                a default bounding box is assigned to a ground truth
                or not. The default value is :obj:`0.5`.

        Returns:
            tuple of two arrays:
            This method returns a tuple of two arrays,
            :obj:`(mb_loc, mb_label)`.

            * **mb_loc**: A float array of shape :math:`(K, 4)`, \
                where :math:`K` is the number of default bounding boxes.
            * **mb_label**: An integer array of shape :math:`(K,)`.

        """
        xp = self.xp

        if len(bbox) == 0:
            return (
                xp.zeros(self._default_bbox.shape, dtype=np.float32),
                xp.zeros(self._default_bbox.shape[:1], dtype=np.int32))

        iou = utils.bbox_iou(
            xp.hstack((
                self._default_bbox[:, :2] - self._default_bbox[:, 2:] / 2,
                self._default_bbox[:, :2] + self._default_bbox[:, 2:] / 2)),
            bbox)
        idx = iou.argmax(axis=1)
        iou = iou.max(axis=1)

        mb_bbox = bbox[idx]
        mb_loc = xp.hstack((
            ((mb_bbox[:, :2] + mb_bbox[:, 2:]) / 2
             - self._default_bbox[:, :2]) /
            (self._variance[0] * self._default_bbox[:, 2:]),
            xp.log((mb_bbox[:, 2:] - mb_bbox[:, :2])
                   / self._default_bbox[:, 2:]) /
            self._variance[1]))

        mb_label = label[idx]
        # [0, n_fg_class - 1] -> [1, n_fg_class]
        mb_label += 1
        # 0 is for background
        mb_label[iou < iou_thresh] = 0

        return mb_loc.astype(np.float32), mb_label.astype(np.int32)

    def decode(self, mb_loc, mb_conf, nms_thresh, score_thresh):
        """Decodes coordinates and classes of bounding boxes.

        This method decodes :obj:`mb_loc` and :obj:`mb_conf` returned
        by a SSD network.

        Args:
            mb_loc (array): A float array whose shape is
                :math:`(K, 4)`, :math:`K` is the number of
                 default bounding boxes.
            mb_conf (array): A float array whose shape is
                :math:`(K, n\_fg\_class + 1)`.
            nms_thresh (float): The threshold value
                for :meth:`chainercv.transfroms.non_maximum_suppression`.
            score_thresh (float): The threshold value for confidence score.
                If a bounding box whose confidence score is lower than
                this value, the bounding box will be suppressed.

        Returns:
            tuple of three arrays:
            This method returns a tuple of three arrays,
            :obj:`(bbox, label, score)`.

            * **bbox**: A float array of shape :math:`(R, 4)`, \
                where :math:`R` is the number of bounding boxes in a image. \
                Each bouding box is organized by \
                :obj:`(x_min, y_min, x_max, y_max)` \
                in the second axis.
            * **label** : An integer array of shape :math:`(R,)`. \
                Each value indicates the class of the bounding box.
            * **score** : A float array of shape :math:`(R,)`. \
                Each value indicates how confident the prediction is.

        """
        xp = self.xp

        # the format of raw_bbox is (center_x, center_y, width, height)
        raw_bbox = xp.hstack((
            self._default_bbox[:, :2] + mb_loc[:, :2] *
            self._variance[0] * self._default_bbox[:, 2:],
            self._default_bbox[:, 2:] *
            xp.exp(mb_loc[:, 2:] * self._variance[1])))
        # convert the format of raw_bbox to (x_min, y_min, x_max, y_max)
        raw_bbox[:, :2] -= raw_bbox[:, 2:] / 2
        raw_bbox[:, 2:] += raw_bbox[:, :2]
        raw_score = xp.exp(mb_conf)
        raw_score /= raw_score.sum(axis=1, keepdims=True)

        bbox = list()
        label = list()
        score = list()
        for l in range(mb_conf.shape[1] - 1):
            bbox_l = raw_bbox
            # the l-th class corresponds for the (l + 1)-th column.
            score_l = raw_score[:, l + 1]

            mask = score_l >= score_thresh
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            if nms_thresh is not None:
                indices = utils.non_maximum_suppression(
                    bbox_l, nms_thresh, score_l)
                bbox_l = bbox_l[indices]
                score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append(xp.array((l,) * len(bbox_l)))
            score.append(score_l)

        bbox = xp.vstack(bbox)
        label = xp.hstack(label).astype(int)
        score = xp.hstack(score)

        return bbox, label, score
