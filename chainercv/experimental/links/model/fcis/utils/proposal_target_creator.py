import numpy as np

from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou


class ProposalTargetCreator(object):
    """Assign ground truth classes, bounding boxes and masks to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train FCIS [#FCIS]_.

    .. [#FCIS] Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei. \
    Fully Convolutional Instance-aware Semantic Segmentation. CVPR 2017.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.
        binary_thresh (float): Threshold for resized mask.

    """

    def __init__(
            self, n_sample=128,
            pos_ratio=0.25, pos_iou_thresh=0.5,
            neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.1,
            binary_thresh=0.4):

        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.binary_thresh = binary_thresh

    def __call__(
            self, roi, mask, label, bbox,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.2, 0.2, 0.5, 0.5),
            mask_size=(21, 21),
    ):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi`, :obj:`mask`, :obj:`label`
        and :obj: `bbox`. The RoIs are assigned with the ground truth class
        labels as well as bounding box offsets and scales to match the ground
        truth bounding boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs
        are sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`chainercv.links.model.faster_rcnn.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.
        * :math:`H` is the image height.
        * :math:`W` is the image width.
        * :math:`RH` is the mask height.
        * :math:`RW` is the mask width.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            mask (array): The coordinates of ground truth masks.
                Its shape is :math:`(R', H, W)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bounding boxes.
            loc_normalize_std (tuple of four floats): Standard deviation of
                the coordinates of bounding boxes.
            mask_size (tuple of int or int): Generated mask size, which is
                equal to :math:`(RH, RW)`.

        Returns:
            (array, array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_mask**: Masks assigned to sampled RoIs. Its shape is \
                :math:`(S, RH, RW)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.

        """

        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        mask = cuda.to_cpu(mask)
        label = cuda.to_cpu(label)
        bbox = cuda.to_cpu(bbox)

        if not isinstance(mask_size, tuple):
            mask_size = (mask_size, mask_size)

        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        if self.n_sample is None:
            n_sample = roi.shape[0]
        else:
            n_sample = self.n_sample

        pos_roi_per_image = np.round(n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both foreground and background).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # locs
        # Compute offsets and scales to match sampled RoIs to the GTs.
        loc_normalize_mean = np.array(loc_normalize_mean, np.float32)
        loc_normalize_std = np.array(loc_normalize_std, np.float32)
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = gt_roi_loc - loc_normalize_mean
        gt_roi_loc = gt_roi_loc / loc_normalize_std

        # masks
        gt_roi_mask = -1 * np.ones(
            (len(keep_index), mask_size[0], mask_size[1]),
            dtype=np.int32)

        for i, pos_ind in enumerate(pos_index):
            bb = np.round(sample_roi[i]).astype(np.int)
            gt_msk = mask[gt_assignment[pos_ind]]
            gt_roi_msk = gt_msk[bb[0]:bb[2], bb[1]:bb[3]]
            gt_roi_msk = resize(
                gt_roi_msk.astype(np.float32)[None], mask_size)[0]
            gt_roi_msk = (gt_roi_msk >= self.binary_thresh).astype(np.int)
            gt_roi_mask[i] = gt_roi_msk

        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            gt_roi_mask = cuda.to_gpu(gt_roi_mask)
            gt_roi_label = cuda.to_gpu(gt_roi_label)
            gt_roi_loc = cuda.to_gpu(gt_roi_loc)

        return sample_roi, gt_roi_mask, gt_roi_label, gt_roi_loc
