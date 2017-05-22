import numpy as np

from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox_regression_target import \
    bbox_regression_target
from chainercv.utils.bbox.bbox_overlap import bbox_overlap


class ProposalTargetCreator(object):
    """Assign proposals to ground-truth targets.

    The :meth:`__call__` of this class generates training targets/labels
    for each object proposal. These are

    * labels to set of :math:`1, ..., K` classes and the background class.
    * Regression targets for bounding boxes in the case when their labels are \
        not the background (i.e. label :math:`> 0`)

    This is used to train Faster RCNN [1].

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_class (int): Number of classes to categorize.
        batch_size (int): Number of regions to produce.
        bbox_normalize_target_precomputed (bool): Normalize the targets
            using :obj:`bbox_normalize_means` and :obj:`bbox_normalize_stds`.
        bbox_normalize_mean (tuple of four floats): Mean values to normalize
            coordinates of bouding boxes.
        bbox_normalize_std (tupler of four floats): Standard deviation of
            the coordinates of bounding boxes.
        bbox_inside_weight (tuple of four floats):
        fg_fraction (float): Fraction of regions that is labeled foreground.
        fg_thresh (float): Overlap threshold for a ROI to be considered
            foreground.
        bg_thresh_hi (float): ROI is considered to be background if overlap is
            in [:obj:`bg_thresh_hi`, :obj:`bg_thresh_hi`).
        bg_thresh_lo (float): See :obj:`bg_thresh_hi`.

    """

    def __init__(self, n_class=21,
                 batch_size=128,
                 bbox_normalize_target_precomputed=True,
                 bbox_normalize_mean=(0., 0., 0., 0.),
                 bbox_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 bbox_inside_weight=(1., 1., 1., 1.),
                 fg_fraction=0.25,
                 fg_thresh=0.5, bg_thresh_hi=0.5, bg_thresh_lo=0.0
                 ):
        self.n_class = n_class
        self.batch_size = batch_size
        self.fg_fraction = fg_fraction
        self.bbox_inside_weight = bbox_inside_weight
        self.bbox_normalize_target_precomputed =\
            bbox_normalize_target_precomputed
        self.bbox_normalize_mean = bbox_normalize_mean
        self.bbox_normalize_std = bbox_normalize_std
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo

    def __call__(self, roi, bbox, label):
        """It assigns labels to sampled proposals from RPN.

        This samples total of :obj:`self.batch_size`. The last
        :obj:`self.fg_fraction * self.batch_size` samples are all
        assigned to the background label.

        Here are notations.

        * :math:`N` is batchsize.
        * :math:`S` nad :math:`R` are numbers of bounding boxes in \
            :obj:`roi` and :obj:`bbox`.
        * :math:`K` is number of object classes.

        Args:
            roi (array): Region of interests from which we sample.
                This is an array whose shape is :math:`(S, 5)`. The
                second axis contains \
                :obj:`(batch_index, x_min, y_min, x_max, y_max)` of \
                each region of interests.
            bbox (array): The ground truth bounding boxes. Its shape is \
                :math:`(N, R, 4)`.
            label (array): The ground truth bounding box labels. Its shape \
                is :math:`(N, R)`.

        Returns:
            (array, array, array, array, array, array):

            * **roi_sample**: Regions of interests that are sampled. \
                This is an array whose shape is \
                :obj:`(self.batch_size, 5)`. The second axis contains \
                :obj:`(batch_index, x_min, y_min, x_max, y_max)` of \
                each region of interests.
            * **bbox_target_sample**: Bounding boxes that are sampled. \
                Its shape is :obj:`(N, self.batch_size, 4K)`. The last \
                axis represents bounding box targets for each of the \
                :math:`K` classes. The coordinates for the same class is \
                contiguous in this array. The coordinates are ordered by \
                :obj:`x_min, y_min, x_max, y_max`.
            * **label_sample**: Labels sampled for training. Its shape is \
                :obj:`(N, self.batch_size)`.
            * **bbox_inside_weight**: Inside weights used to compute losses \
                for Faster RCNN. Its shape is \
                :math:`(N, self.batch_size, 4K)`. The last axis is organized \
                similarly to :obj:`bbox_target_sample`.
            * **bbox_outside_weight**: Outside weights used to compute losses \
                for Faster RCNN. Its shape is \
                :math:`(N, self.batch_size, 4K)`. The last axis is organized \
                similarly to :obj:`bbox_target_sample`.

        """
        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        bbox = cuda.to_cpu(bbox)
        label = cuda.to_cpu(label)

        assert bbox.ndim == 3
        n_image, n_bbox, _ = bbox.shape
        assert bbox.shape[0] == 1
        assert label.shape[0] == 1
        assert np.all(roi[:, 0] == 0), 'Only single item batches are supported'

        bbox = bbox[0]
        label = label[0]

        gt_roi = np.hstack((np.zeros((n_bbox, 1), dtype=bbox.dtype), bbox))
        roi = np.vstack((roi, gt_roi))

        rois_per_image = self.batch_size // n_image
        fg_rois_per_image = np.round(self.fg_fraction * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        label_sample, roi_sample, bbox_target_sample, bbox_inside_weight =\
            self._sample_roi(
                roi, bbox, label, fg_rois_per_image,
                rois_per_image, self.n_class)
        label_sample = label_sample.astype(np.int32)
        roi_sample = roi_sample.astype(np.float32)

        bbox_outside_weight = (bbox_inside_weight > 0).astype(np.float32)

        if xp != np:
            roi_sample = cuda.to_gpu(roi_sample)
            bbox_target_sample = cuda.to_gpu(bbox_target_sample)
            label_sample = cuda.to_gpu(label_sample)
            bbox_inside_weight = cuda.to_gpu(bbox_inside_weight)
            bbox_outside_weight = cuda.to_gpu(bbox_outside_weight)
        return roi_sample, bbox_target_sample[None], label_sample[None],\
            bbox_inside_weight[None], bbox_outside_weight[None]

    def _sample_roi(
            self, roi, bbox, label, fg_rois_per_image, rois_per_image,
            n_class):
        # Generate a random sample of RoIs comprising foreground and background
        # examples.
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlap(roi[:, 1:5], bbox)
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        label_sample = label[gt_assignment]

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= self.fg_thresh)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = np.random.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < self.bg_thresh_hi) &
                           (max_overlaps >= self.bg_thresh_lo))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = np.random.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        # Select sampled values from various arrays:
        label_sample = label_sample[keep_inds]
        # Clamp labels for the background RoIs to 0
        label_sample[fg_rois_per_this_image:] = 0
        roi_sample = roi[keep_inds]

        bbox_sample = bbox_regression_target(
            roi_sample[:, 1:5], bbox[gt_assignment[keep_inds]])
        if self.bbox_normalize_target_precomputed:
            # Optionally normalize targets by a precomputed mean and stdev
            bbox_sample = ((bbox_sample - np.array(self.bbox_normalize_mean)
                            ) / np.array(self.bbox_normalize_std))

        bbox_target_sample, bbox_inside_weight = \
            _get_bbox_regression_label(
                bbox_sample, label_sample, n_class, self.bbox_inside_weight)
        return label_sample, roi_sample, bbox_target_sample, bbox_inside_weight


def _get_bbox_regression_label(bbox, label, n_class, bbox_inside_weight_coeff):
    # Bounding-box regression targets (bbox_target_data) are stored in a
    # compact form N x (class, tx, ty, tw, th)
    # This function expands those targets into the 4-of-4*K representation
    # used by the network (i.e. only one class has non-zero targets).

    # Returns:
    #     bbox_target (ndarray): N x 4K blob of regression targets
    #     bbox_inside_weights (ndarray): N x 4K blob of loss weights

    n_bbox = label.shape[0]
    bbox_target = np.zeros((n_bbox, 4 * n_class), dtype=np.float32)
    bbox_inside_weight = np.zeros_like(bbox_target)
    inds = np.where(label > 0)[0]
    for ind in inds:
        cls = int(label[ind])
        start = int(4 * cls)
        end = int(start + 4)
        bbox_target[ind, start:end] = bbox[ind]
        bbox_inside_weight[ind, start:end] = bbox_inside_weight_coeff
    return bbox_target, bbox_inside_weight
