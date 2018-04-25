import numpy as np

from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils import non_maximum_suppression


def _mask_aggregation(
        bbox, mask_prob, mask_weight,
        size, binary_thresh
):
    assert bbox.shape[0] == len(mask_prob)
    assert bbox.shape[0] == mask_weight.shape[0]

    mask = np.zeros(size, dtype=np.float32)
    for bb, msk_pb, msk_w in zip(bbox, mask_prob, mask_weight):
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        if y_max - y_min > 0 and x_max - x_min > 0:
            msk_pb = resize(
                msk_pb.astype(np.float32)[None],
                (y_max - y_min, x_max - x_min))
            msk_m = (msk_pb >= binary_thresh).astype(np.float32)[0]
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
        roi_mask_prob, bbox, cls_prob, size,
        score_thresh, nms_thresh,
        mask_merge_thresh, binary_thresh,
        limit=100, bg_label=0
):
    """Refine mask probabilities by merging multiple masks.

    First, this function discard invalid masks with non maximum suppression.
    Then, it merges masks with weight calculated from class probabilities and
    iou.
    This function improves the mask qualities by merging overlapped masks
    predicted as the same object class.

    Here are notations used.
    * :math:`R'` is the total number of RoIs produced across batches.
    * :math:`L` is the number of classes excluding the background.
    * :math:`RH` is the height of pooled image.
    * :math:`RW` is the height of pooled image.

    Args:
        roi_mask_prob (array): A mask probability array whose shape is
            :math:`(R, RH, RW)`.
        bbox (array): A bounding box array whose shape is
            :math:`(R, 4)`.
        cls_prob (array): A class probability array whose shape is
            :math:`(R, L + 1)`.
        size (tuple of int): Original image size.
        score_thresh (float): A threshold value of the class score.
        nms_thresh (float): A threshold value of non maximum suppression.
        mask_merge_thresh (float): A threshold value of the bounding box iou
            for mask merging.
        binary_thresh (float): A threshold value of mask score
            for mask merging.
        limit (int): The maximum number of outputs.
        bg_label (int): The id of the background label.

    Returns:
        array, array, array, array:
        * **v_mask_score**: Merged masks. Its shapes is :math:`(N, RH, RW)`.
        * **v_bbox**: Bounding boxes for the merged masks. Its shape is \
            :math:`(N, 4)`.
        * **v_label**: Class labels for the merged masks. Its shape is \
            :math:`(N, )`.
        * **v_score**: Class probabilities for the merged masks. Its shape \
            is :math:`(N, )`.

    """

    roi_mask_size = roi_mask_prob.shape[1:]
    n_class = cls_prob.shape[1]

    v_mask_prob = []
    v_bbox = []
    v_label = []
    v_cls_prob = []

    cls_score = []
    cls_bbox = []

    for label in range(0, n_class):
        # background
        if label == bg_label:
            continue
        # non maximum suppression
        score_l = cls_prob[:, label]
        keep_indices = non_maximum_suppression(
            bbox, nms_thresh, score_l)
        bbox_l = bbox[keep_indices]
        score_l = score_l[keep_indices]
        cls_bbox.append(bbox_l)
        cls_score.append(score_l)

    sorted_score = np.sort(np.concatenate(cls_score))[::-1]
    n_keep = min(len(sorted_score), limit)
    score_thresh = max(sorted_score[n_keep - 1], score_thresh)

    for label in range(0, n_class):
        # background
        if label == bg_label:
            continue
        bbox_l = cls_bbox[label - 1]
        score_l = cls_score[label - 1]
        keep_indices = np.where(score_l >= score_thresh)
        bbox_l = bbox_l[keep_indices]
        score_l = score_l[keep_indices]

        v_mask_prob_l = []
        v_bbox_l = []
        v_score_l = []

        for i, bb in enumerate(bbox_l):
            iou = bbox_iou(bbox, bb[np.newaxis, :])
            keep_indices = np.where(iou >= mask_merge_thresh)[0]
            mask_weight = cls_prob[keep_indices, label]
            mask_weight = mask_weight / mask_weight.sum()
            mask_prob_i = roi_mask_prob[keep_indices]
            bbox_i = bbox[keep_indices]
            c_mask, c_bbox = _mask_aggregation(
                bbox_i, mask_prob_i, mask_weight, size, binary_thresh)
            if c_mask is not None and c_bbox is not None:
                c_mask = resize(
                    c_mask.astype(np.float32),
                    roi_mask_size)
                v_mask_prob_l.append(c_mask)
                v_bbox_l.append(c_bbox)
                v_score_l.append(score_l[i])

        if len(v_mask_prob_l) > 0:
            v_label_l = np.repeat(
                label - 1, len(v_score_l)).astype(np.int32)

            v_mask_prob += v_mask_prob_l
            v_bbox += v_bbox_l
            v_label.append(v_label_l)
            v_cls_prob.append(v_score_l)

    if len(v_mask_prob) > 0:
        v_mask_prob = np.concatenate(v_mask_prob)
        v_bbox = np.concatenate(v_bbox)
        v_label = np.concatenate(v_label)
        v_cls_prob = np.concatenate(v_cls_prob)
    else:
        v_mask_prob = np.empty((0, roi_mask_size[0], roi_mask_size[1]))
        v_bbox = np.empty((0, 4))
        v_label = np.empty((0, ))
        v_cls_prob = np.empty((0, ))
    return v_mask_prob, v_bbox, v_label, v_cls_prob
