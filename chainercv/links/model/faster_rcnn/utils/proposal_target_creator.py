import numpy as np

from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.utils.bbox.bbox_iou import bbox_iou


class ProposalTargetCreator(object):
    """Assign proposals to ground-truth targets.

    The :meth:`__call__` of this class generates training targets/labels
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_class (int): Number of classes possibly including the background.
        batch_size (int): Number of regions to produce.
        loc_normalize_mean (tuple of four floats): Mean values to normalize
            coordinates of bouding boxes.
        loc_normalize_std (tupler of four floats): Standard deviation of
            the coordinates of bounding boxes.
        loc_in_weight (tuple of four floats): Weights applied to
            :obj:`loc` used by Faster R-CNN.
        fg_fraction (float): Fraction of regions that is labeled foreground.
        fg_thresh (float): IoU threshold for a ROI to be considered
            foreground.
        bg_thresh_hi (float): ROI is considered to be background if IoU is
            in [:obj:`bg_thresh_hi`, :obj:`bg_thresh_hi`).
        bg_thresh_lo (float): See above.

    """

    def __init__(self, n_class,
                 batch_size=128,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 loc_in_weight=(1., 1., 1., 1.),
                 fg_fraction=0.25,
                 fg_thresh=0.5, bg_thresh_hi=0.5, bg_thresh_lo=0.0
                 ):
        self.n_class = n_class
        self.batch_size = batch_size
        self.fg_fraction = fg_fraction
        self.loc_in_weight = loc_in_weight
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo

    def __call__(self, roi, bbox, label):
        """Assigns labels to sampled proposals from RPN.

        This samples total of :obj:`self.batch_size` RoIs from concatenated
        list of bounding boxes from :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels and bounding
        box offsets.
        As many as :obj:`fg_fraction * self.batch_size` RoIs are
        sampled with foreground label assignments.

        The second axis of the bounding box arrays contain coordinates
        of bounding boxes which are ordered by
        :obj:`(x_min, y_min, x_max, y_max)`.
        Offsets of bounding boxes are calculated using
        :func:`chainercv.links.model.faster_rcnn.bbox2loc`.
        Also, types of inputs and outputs are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.batch_size`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of interests from which we sample.
                This is an array whose shape is :math:`(R, 4)`
            bbox (array): The ground truth bounding boxes. Its shape is \
                :math:`(R', 4)`.
            label (array): The ground truth bounding box labels. Its shape \
                is :math:`(R',)`.

        Returns:
            (array, array, array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_cls_loc**: Ground truth offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, L \\times 4)`. The last \
                axis represents bounding box offsets for each of the \
                :math:`L` classes. The coordinates for the same class is \
                contiguous in this array.
            * **gt_roi_label**: Labels sampled for training. Its shape is \
                :math:`(S,)`.
            * **roi_loc_in_weight**: Inside weights used to \
                compute losses for Faster RCNN. Its shape is \
                :math:`(S, L \\times 4)`. The second axis is organized \
                similarly to :obj:`gt_roi_cls_loc`.
            * **roi_loc_out_weight**: Outside weights used to compute \
                losses for Faster RCNN. Its shape is \
                :math:`(S, L \\times 4)`. The second axis is organized \
                similarly to :obj:`gt_roi_cls_loc`.

        """
        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        bbox = cuda.to_cpu(bbox)
        label = cuda.to_cpu(label)

        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        # Sample rois with classification labels and offsets/scales to the GTs.
        sample_roi, gt_roi_loc, gt_roi_label = self._sample_roi(
            roi, bbox, label)

        # Convert loc (R, 4) and cls (R,) to obtain cls_loc (R, L * 4)
        gt_roi_cls_loc, roi_loc_in_weight =\
            self._get_bbox_regression_label(
                gt_roi_loc, gt_roi_label, self.n_class)

        roi_loc_out_weight = (roi_loc_in_weight > 0).astype(np.float32)

        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            gt_roi_cls_loc = cuda.to_gpu(gt_roi_cls_loc)
            gt_roi_label = cuda.to_gpu(gt_roi_label)
            roi_loc_in_weight = cuda.to_gpu(roi_loc_in_weight)
            roi_loc_out_weight = cuda.to_gpu(roi_loc_out_weight)
        return sample_roi, gt_roi_cls_loc, gt_roi_label,\
            roi_loc_in_weight, roi_loc_out_weight

    def _sample_roi(self, roi, bbox, label):
        fg_roi_per_image = np.round(self.batch_size * self.fg_fraction)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment]

        # Select foreground RoIs as those with >= FG_THRESH IoU.
        fg_index = np.where(max_iou >= self.fg_thresh)[0]
        fg_roi_per_this_image = int(min(fg_roi_per_image, fg_index.size))
        if fg_index.size > 0:
            fg_index = np.random.choice(
                fg_index, size=fg_roi_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI).
        bg_index = np.where((max_iou < self.bg_thresh_hi) &
                            (max_iou >= self.bg_thresh_lo))[0]
        bg_roi_per_this_image = self.batch_size - fg_roi_per_this_image
        bg_roi_per_this_image = int(min(bg_roi_per_this_image, bg_index.size))
        if bg_index.size > 0:
            bg_index = np.random.choice(
                bg_index, size=bg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg).
        keep_index = np.append(fg_index, bg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[fg_roi_per_this_image:] = 0  # BG labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(self.loc_normalize_mean)
                       ) / np.array(self.loc_normalize_std))
        return sample_roi, gt_roi_loc, gt_roi_label

    def _get_bbox_regression_label(self, loc, label, n_class):
        # From loc (R, 4) and label (R,), this function computes
        # cls_loc (R, L * 4).
        # Only one class has non-zero targets in this representation.

        n_bbox = label.shape[0]
        cls_loc = np.zeros((n_bbox, 4 * n_class), dtype=np.float32)
        loc_in_weight = np.zeros_like(cls_loc)
        index = np.where(label > 0)[0]
        for ind in index:
            l = int(label[ind])
            start = int(4 * l)
            end = int(start + 4)
            cls_loc[ind, start:end] = loc[ind]
            loc_in_weight[ind, start:end] = self.loc_in_weight
        return cls_loc, loc_in_weight
