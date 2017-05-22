import numpy as np

import chainer
from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox_regression_target import \
    bbox_regression_target
from chainercv.utils.bbox.bbox_overlap import bbox_overlap


class AnchorTargetCreator(object):

    """Assign anchors to ground-truth targets.

    It produces followings.

    * labels anchors to either of foreground, background and ignore.
    * bounding-box regression targets.

    These supervising data is used to train networks such as Region
    Proposal Networks introduced in Faster RCNN [1].

    Bounding regression targets are computed using encoding scheme
    found in :obj:`chainercv.links.utils.bbox_regression_target`.

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        rpn_batchsize (int): Number of regions to produce.
        rpn_negative_overlap (float): Anchors with overlap below this
            threshold will be assigned as negative.
        rpn_positive_overlap (float): Anchors with overlap above this
            threshold will be assigned as positive.
        rpn_fg_fraction (float): Fraction of positive regions in the
            set of all regions produced.
        rpn_bbox_inside_weight (tuple of four floats): Four coefficients
            used to calculate bbox_inside_weight.

    .. seealso::
        :obj:`chainercv.links.utils.bbox_regression_target`

    """

    def __init__(self,
                 rpn_batchsize=256,
                 rpn_negative_overlap=0.3, rpn_positive_overlap=0.7,
                 rpn_fg_fraction=0.5,
                 rpn_bbox_inside_weight=(1., 1., 1., 1.)):
        self.rpn_batchsize = rpn_batchsize
        self.rpn_negative_overlap = rpn_negative_overlap
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_fg_fraction = rpn_fg_fraction
        self.rpn_bbox_inside_weight = rpn_bbox_inside_weight

    def __call__(self, bbox, anchor, feat_size, img_size):
        """Calculate targets of classification labels and bbox regressions.

        Here are notations.

        * :math:`A` is number of anchors.
        * :math:`R` is number of bounding boxes.
        * :math:`H` and `W` are height and width of features.
        * :math:`N` is batch size.

        For arrays of bounding boxes, its second axis contains
        x and y coordinates of left top vertices and right bottom vertices.

        Args:
            bbox (array): An array of shape :math:`(R, 4)`.
            anchor (array): An array of shape :math:`(A, 4)`.
            feat_size (tuple of ints): A tuple :obj:`W, H`.
            img_size (tuple of ints): A tuple :obj:`img_W, img_H`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array, array, array):

            Tuple of four arrays which contains the following elements.

            * **bbox_target**: Bounding boxes encoded into regression \
                targets. This is an array of shape :math:`(N, 4 A, H, W)`.
            * **label**: Labels of bounding boxes with values \
                :obj:`(1=foreground, 0=background, -1=ignore)`. Its shape \
                is :math:`(N, A, H, W)`.
            * **bbox_inside_weight**: Inside weight used to compute losses \
                for Faster RCNN. Its shape is :math:`(N, 4 A, H, W)`.
            * **bbox_outside_weight** Outside weight used to compute losses \
                for Faster RCNN. Its shape is :math:`(N, 4 A, H, W)`.

        """
        assert bbox.ndim == 3
        assert bbox.shape[0] == 1
        if isinstance(bbox, chainer.Variable):
            bbox = bbox.data
        xp = cuda.get_array_module(bbox)
        bbox = cuda.to_cpu(bbox)
        anchor = cuda.to_cpu(anchor)

        bbox = bbox[0]

        width, height = feat_size
        img_W, img_H = img_size

        n_anchor = len(anchor)
        inds_inside, anchor = _keep_inside(anchor, img_W, img_H)
        argmax_overlaps, label = self._create_label(
            inds_inside, anchor, bbox)

        # compute bounding box regression targets
        bbox_target = bbox_regression_target(anchor, bbox[argmax_overlaps])

        # calculate inside and outside weights weights
        bbox_inside_weight = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weight[label == 1, :] = np.array(
            self.rpn_bbox_inside_weight)
        bbox_outside_weight = self._calc_outside_weights(inds_inside, label)

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inds_inside, fill=-1)
        bbox_target = _unmap(
            bbox_target, n_anchor, inds_inside, fill=0)
        bbox_inside_weight = _unmap(
            bbox_inside_weight, n_anchor, inds_inside, fill=0)
        bbox_outside_weight = _unmap(
            bbox_outside_weight, n_anchor, inds_inside, fill=0)

        # reshape
        bbox_target = bbox_target.reshape(
            (1, height, width, -1)).transpose(0, 3, 1, 2)
        label = label.reshape(
            (1, height, width, -1)).transpose(0, 3, 1, 2)
        label = label.astype(np.int32)
        bbox_inside_weight = bbox_inside_weight.reshape(
            (1, height, width, -1)).transpose(0, 3, 1, 2)
        bbox_outside_weight = bbox_outside_weight.reshape(
            (1, height, width, -1)).transpose(0, 3, 1, 2)

        if xp != np:
            bbox_target = chainer.cuda.to_gpu(bbox_target)
            label = chainer.cuda.to_gpu(label)
            bbox_inside_weight = chainer.cuda.to_gpu(
                bbox_inside_weight)
            bbox_outside_weight = chainer.cuda.to_gpu(
                bbox_outside_weight)
        return bbox_target, label, bbox_inside_weight, bbox_outside_weight

    def _create_label(self, inds_inside, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inds_inside), ), dtype=np.float32)
        label.fill(-1)

        argmax_overlaps, max_overlaps, gt_max_overlaps, gt_argmax_overlaps = \
            self._calc_overlaps(anchor, bbox, inds_inside)

        # assign bg labels first so that positive labels can clobber them
        label[max_overlaps < self.rpn_negative_overlap] = 0

        # fg label: for each gt, anchor with highest overlap
        label[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        label[max_overlaps >= self.rpn_positive_overlap] = 1

        # subsample positive labels if we have too many
        num_fg = int(self.rpn_fg_fraction * self.rpn_batchsize)
        fg_inds = np.where(label == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            label[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.rpn_batchsize - np.sum(label == 1)
        bg_inds = np.where(label == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            label[disable_inds] = -1

        return argmax_overlaps, label

    def _calc_overlaps(self, anchor, bbox, inds_inside):
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlap(
            np.ascontiguousarray(anchor, dtype=np.float),
            np.ascontiguousarray(bbox, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        return argmax_overlaps, max_overlaps, gt_max_overlaps, \
            gt_argmax_overlaps

    def _calc_outside_weights(self, inds_inside, label):
        bbox_outside_weight = np.zeros(
            (len(inds_inside), 4), dtype=np.float32)
        # uniform weighting of examples (given non-uniform sampling)
        n_example = np.sum(label >= 0)

        positive_weight = np.ones((1, 4)) * 1.0 / n_example
        negative_weight = np.ones((1, 4)) * 1.0 / n_example

        bbox_outside_weight[label == 1, :] = positive_weight
        bbox_outside_weight[label == 0, :] = negative_weight

        return bbox_outside_weight


def _unmap(data, count, inds, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _keep_inside(anchor, W, H):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    xp = cuda.get_array_module(anchor)

    index_inside = xp.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] < W) &  # width
        (anchor[:, 3] < H)  # height
    )[0]
    return index_inside, anchor[index_inside]
