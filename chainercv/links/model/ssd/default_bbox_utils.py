from __future__ import division

import itertools
import numpy as np

import chainer

from chainercv import utils


def generate_default_bbox(grids, aspect_ratios, steps, sizes):
    """Generate a set of default bounding boxes.

    This function generates a set of default bounding boxes
    which is used in Single Shot Multibox Detector [#]_.

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

    Returns:
        ~numpy.ndarray:
        An array whose shape is :math:`(K, 4)`, where :math:`K` is
        the number of default bounding boxes. Each bounding box is
        organized by :obj:`(center_x, center_y, width, height)`.
    """

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

    default_bbox = np.stack(default_bbox)
    return default_bbox


def encode_with_default_bbox(
        bbox, label, default_bbox, variance, thresh):
    if len(bbox) == 0:
        return (
            np.zeros(default_bbox.shape, dtype=np.float32),
            np.zeros(default_bbox.shape[:1], dtype=np.int32))

    iou = utils.bbox_iou(
        # convert the format of bbox to (x_min, y_min, x_max, y_max)
        np.hstack((
            default_bbox[:, :2] - default_bbox[:, 2:] / 2,
            default_bbox[:, :2] + default_bbox[:, 2:] / 2)),
        bbox)

    gt_idx = iou.argmax(axis=1)
    bbox = bbox[gt_idx]
    label = label[gt_idx]

    loc = np.hstack((
        ((bbox[:, :2] + bbox[:, 2:]) / 2 - default_bbox[:, :2]) /
        (variance[0] * default_bbox[:, 2:]),
        np.log(
            (bbox[:, 2:] - bbox[:, :2]) / default_bbox[:, 2:]) / variance[1]))

    # add 1 to shift the label
    conf = label + 1
    # if IoU is less than thresh, the bounding box is assigned to background
    conf[iou.max(axis=1) < thresh] = 0

    return loc.astype(np.float32), conf.astype(np.int32)


def decode_with_default_bbox(
        loc, conf, default_bbox, variance, nms_thresh, score_thresh):
    """Decode coordinates and classes of bounding boxes.

    This function decodes :obj:`loc` and :obj:`conf` returned
    by a SSD network.

    Args:
        loc (array): A float array whose shape is
            :math:`(K, 4)`, :math:`K` is the number of default bounding boxes.
        conf (array): A float array whose shape is
            :math:`(K, n\_fg\_class + 1)`.
        default_bbox (array): An array holding coordinates of the default
            bounding boxes. Its shape is :math:`(K, 4)`. Each bounding box is
            organized by :obj:`(center_x, center_y, width, height)`.
        variance (tuple of floats): Two coefficients for decoding
            the locations of bounding boxe. The first value is used to
            decode coordinates of the centers. The second value is used to
            decode the sizes of bounding boxes.
        nms_thresh (float): The threshold value
            for :meth:`chainercv.transfroms.non_maximum_suppression`.
        score_thresh (float): The threshold value for confidence score.
            If a bounding box whose confidence score is lower than this value,
            the bounding box will be suppressed.

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

    xp = chainer.cuda.get_array_module(loc, conf, default_bbox)

    # the format of raw_bbox is (center_x, center_y, width, height)
    raw_bbox = xp.hstack((
        default_bbox[:, :2] + loc[:, :2] * variance[0] * default_bbox[:, 2:],
        default_bbox[:, 2:] * xp.exp(loc[:, 2:] * variance[1])))
    # convert the format of raw_bbox to (x_min, y_min, x_max, y_max)
    raw_bbox[:, :2] -= raw_bbox[:, 2:] / 2
    raw_bbox[:, 2:] += raw_bbox[:, :2]
    raw_score = xp.exp(conf)
    raw_score /= raw_score.sum(axis=1, keepdims=True)

    bbox = list()
    label = list()
    score = list()
    for l in range(conf.shape[1] - 1):
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
