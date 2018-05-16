import numpy as np

from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils import non_maximum_suppression


def _mask_aggregation(
        bbox, cmask_prob, cmask_weight,
        size, binary_thresh
):
    assert bbox.shape[0] == len(cmask_prob)
    assert bbox.shape[0] == cmask_weight.shape[0]

    aggregated_msk = np.zeros(size, dtype=np.float32)
    for bb, cmsk_pb, cmsk_w in zip(bbox, cmask_prob, cmask_weight):
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        if y_max - y_min > 0 and x_max - x_min > 0:
            cmsk_pb = resize(
                cmsk_pb.astype(np.float32)[None],
                (y_max - y_min, x_max - x_min))
            cmsk_m = (cmsk_pb >= binary_thresh).astype(np.float32)[0]
            aggregated_msk[y_min:y_max, x_min:x_max] += cmsk_m * cmsk_w

    y_indices, x_indices = np.where(aggregated_msk >= binary_thresh)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None, None
    else:
        y_max = y_indices.max() + 1
        y_min = y_indices.min()
        x_max = x_indices.max() + 1
        x_min = x_indices.min()

        aggregated_bb = np.array(
            [y_min, x_min, y_max, x_max],
            dtype=np.float32)
        aggregated_cmsk = aggregated_msk[y_min:y_max, x_min:x_max]
        return aggregated_cmsk[None], aggregated_bb[None]


def mask_voting(
        roi_cmask_prob, bbox, roi_cls_prob, size,
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
        roi_cmask_prob (array): A mask probability array whose shape is
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
        * **v_cmask_prob**: Merged mask probability. Its shapes is \
            :math:`(N, RH, RW)`.
        * **v_bbox**: Bounding boxes for the merged masks. Its shape is \
            :math:`(N, 4)`.
        * **v_label**: Class labels for the merged masks. Its shape is \
            :math:`(N, )`.
        * **v_score**: Class probabilities for the merged masks. Its shape \
            is :math:`(N, )`.

    """

    roi_cmask_size = roi_cmask_prob.shape[1:]
    n_class = roi_cls_prob.shape[1]

    v_cmask_prob = []
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
        score_l = roi_cls_prob[:, label]
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

        v_cmask_prob_l = []
        v_bbox_l = []
        v_score_l = []

        for i, bb in enumerate(bbox_l):
            iou = bbox_iou(bbox, bb[np.newaxis, :])
            keep_indices = np.where(iou >= mask_merge_thresh)[0]
            cmask_weight = roi_cls_prob[keep_indices, label]
            cmask_weight = cmask_weight / cmask_weight.sum()
            cmask_prob_i = roi_cmask_prob[keep_indices]
            bbox_i = bbox[keep_indices]
            m_cmask, m_bbox = _mask_aggregation(
                bbox_i, cmask_prob_i, cmask_weight, size, binary_thresh)
            if m_cmask is not None and m_bbox is not None:
                m_cmask = resize(
                    m_cmask.astype(np.float32),
                    roi_cmask_size)
                m_cmask = np.clip(m_cmask, 0.0, 1.0)
                v_cmask_prob_l.append(m_cmask)
                v_bbox_l.append(m_bbox)
                v_score_l.append(score_l[i])

        if len(v_cmask_prob_l) > 0:
            v_cmask_prob_l = np.concatenate(v_cmask_prob_l)
            v_bbox_l = np.concatenate(v_bbox_l)
            v_score_l = np.array(v_score_l)

            v_label_l = np.repeat(label - 1, v_bbox_l.shape[0])
            v_label_l = v_label_l.astype(np.int32)
            v_cmask_prob.append(v_cmask_prob_l)
            v_bbox.append(v_bbox_l)
            v_label.append(v_label_l)
            v_cls_prob.append(v_score_l)

    if len(v_cmask_prob) > 0:
        v_cmask_prob = np.concatenate(v_cmask_prob)
        v_bbox = np.concatenate(v_bbox)
        v_label = np.concatenate(v_label)
        v_cls_prob = np.concatenate(v_cls_prob)
    else:
        v_cmask_prob = np.empty((0, roi_cmask_size[0], roi_cmask_size[1]))
        v_bbox = np.empty((0, 4))
        v_label = np.empty((0, ))
        v_cls_prob = np.empty((0, ))
    return v_cmask_prob, v_bbox, v_label, v_cls_prob
