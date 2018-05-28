import numpy as np

from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou


def _generate_gt_roi_msk(gt_msk, gt_bb, bb, mask_size):
    y_min = max(bb[0], gt_bb[0])
    x_min = max(bb[1], gt_bb[1])
    y_max = min(bb[2], gt_bb[2])
    x_max = min(bb[3], gt_bb[3])

    if y_min > y_max or x_min > x_max:
        return np.zeros((mask_size, mask_size))

    h = y_max - y_min
    w = x_max - x_min
    y_start = y_min - bb[0]
    x_start = x_min - bb[1]
    y_end = y_start + h
    x_end = x_start + w

    gt_roi_msk = np.zeros(
        (bb[2] - bb[0], bb[3] - bb[1]), dtype=np.float32)
    gt_clipped_msk = gt_msk[y_min:y_max, x_min:x_max]
    gt_roi_msk[y_start:y_end, x_start:x_end] = gt_clipped_msk
    return gt_roi_msk


class ProposalTargetCreator(object):
    def __init__(
            self, n_sample=128,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.2, 0.2, 0.5, 0.5),
            pos_ratio=0.25, pos_iou_thresh=0.5,
            neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
            mask_size=21, binary_thresh=0.4):

        self.n_sample = n_sample
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

    def __call__(self, roi, bbox, mask, label):

        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        bbox = cuda.to_cpu(bbox)
        mask = cuda.to_cpu(mask)
        label = cuda.to_cpu(label)

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
        loc_normalize_mean = np.array(self.loc_normalize_mean, np.float32)
        loc_normalize_std = np.array(self.loc_normalize_std, np.float32)
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = gt_roi_loc - loc_normalize_mean
        gt_roi_loc = gt_roi_loc / loc_normalize_std

        # masks
        gt_roi_mask = -1 * np.ones(
            (len(keep_index), self.mask_size, self.mask_size),
            dtype=np.int32)

        for i, pos_ind in enumerate(pos_index):
            bb = np.round(sample_roi[i]).astype(np.int)
            gt_bb = np.round(bbox[gt_assignment[pos_ind]]).astype(np.int)
            gt_msk = mask[gt_assignment[pos_ind]]
            gt_roi_msk = _generate_gt_roi_msk(
                gt_msk, gt_bb, bb, self.mask_size)
            gt_roi_msk = resize(
                gt_roi_msk[None], (self.mask_size, self.mask_size))[0]
            gt_roi_msk = (gt_roi_msk >= self.binary_thresh).astype(np.int)
            gt_roi_mask[i] = gt_roi_msk

        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            gt_roi_loc = cuda.to_gpu(gt_roi_loc)
            gt_roi_mask = cuda.to_gpu(gt_roi_mask)
            gt_roi_label = cuda.to_gpu(gt_roi_label)

        return sample_roi, gt_roi_loc, gt_roi_mask, gt_roi_label
