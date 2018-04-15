import numpy as np

from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils import non_maximum_suppression


def mask_aggregation(
        bbox, mask_prob, mask_weight,
        size, binary_thresh
):
    assert bbox.shape[0] == len(mask_prob)
    assert bbox.shape[0] == mask_weight.shape[0]

    mask = np.zeros(size, dtype=np.float32)
    for bb, msk_prb, msk_w in zip(bbox, mask_prob, mask_weight):
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        msk_prb = resize(
            msk_prb.astype(np.float32)[None], (y_max - y_min, x_max - x_min))
        msk_m = (msk_prb >= binary_thresh).astype(np.float32)[0]
        mask[y_min:y_max, x_min:x_max] += msk_m * msk_w

    y_indices, x_indices = np.where(mask >= binary_thresh)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None, None
    else:
        y_max = y_indices.max() + 1
        y_min = y_indices.min()
        x_max = x_indices.max() + 1
        x_min = x_indices.min()

        c_bbox = np.array(
            [y_min, x_min, y_max, x_max],
            dtype=np.float32)
        c_mask = mask[y_min:y_max, x_min:x_max]
        return c_mask[None], c_bbox[None]


def mask_voting(
        mask_prob, prob, bbox,
        size, n_class,
        score_thresh, nms_thresh,
        mask_merge_thresh, binary_thresh,
        limit=100, bg_label=0
):

    mask_size = mask_prob.shape[-1]
    v_mask_prob = []
    v_label = []
    v_prob = []
    v_bbox = []

    cls_prob = []
    cls_bbox = []

    for label in range(0, n_class):
        # background
        if label == bg_label:
            continue
        # non maximum suppression
        prob_l = prob[:, label]
        keep_indices = non_maximum_suppression(
            bbox, nms_thresh, prob_l)
        bbox_l = bbox[keep_indices]
        prob_l = prob_l[keep_indices]
        cls_bbox.append(bbox_l)
        cls_prob.append(prob_l)

    sorted_prob = np.sort(np.concatenate(cls_prob))[::-1]
    keep_n = min(len(sorted_prob), limit)
    thresh = max(sorted_prob[keep_n - 1], 1e-3)

    for label in range(0, n_class):
        # background
        if label == bg_label:
            continue
        bbox_l = cls_bbox[label - 1]
        prob_l = cls_prob[label - 1]
        keep_indices = np.where(prob_l >= thresh)
        bbox_l = bbox_l[keep_indices]
        prob_l = prob_l[keep_indices]

        v_mask_prob_l = []
        v_prob_l = []
        v_bbox_l = []

        for i, bb in enumerate(bbox_l):
            iou = bbox_iou(bbox, bb[np.newaxis, :])
            keep_indices = np.where(iou >= mask_merge_thresh)[0]
            mask_weight = prob[keep_indices, label]
            mask_weight = mask_weight / mask_weight.sum()
            mask_prob_i = mask_prob[keep_indices]
            bbox_i = bbox[keep_indices]
            c_mask, c_bbox = mask_aggregation(
                bbox_i, mask_prob_i, mask_weight, size, binary_thresh)
            if c_mask is not None and c_bbox is not None:
                c_mask = resize(
                    c_mask.astype(np.float32),
                    (mask_size, mask_size))
                v_mask_prob_l.append(c_mask)
                v_bbox_l.append(c_bbox)
                v_prob_l.append(prob_l[i])

        if len(v_mask_prob_l) > 0:
            v_mask_prob_l = np.concatenate(v_mask_prob_l)
            v_prob_l = np.array(v_prob_l)
            v_bbox_l = np.concatenate(v_bbox_l)

            keep_indices = v_prob_l > score_thresh
            v_mask_prob_l = v_mask_prob_l[keep_indices]
            v_prob_l = v_prob_l[keep_indices]
            v_bbox_l = v_bbox_l[keep_indices]

            v_label_l = np.repeat(label - 1, v_bbox_l.shape[0])
            v_label_l = v_label_l.astype(np.int32)
            v_mask_prob.append(v_mask_prob_l)
            v_label.append(v_label_l)
            v_prob.append(v_prob_l)
            v_bbox.append(v_bbox_l)

    if len(v_mask_prob) > 0:
        v_mask_prob = np.concatenate(v_mask_prob)
        v_label = np.concatenate(v_label)
        v_prob = np.concatenate(v_prob)
        v_bbox = np.concatenate(v_bbox)
    else:
        v_mask_prob = np.empty((0, size[0], size[1]))
        v_label = np.empty((0, ))
        v_prob = np.empty((0, ))
        v_bbox = np.empty((0, 4))
    return v_mask_prob, v_label, v_prob, v_bbox
