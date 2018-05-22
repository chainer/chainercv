import numpy as np

from chainercv.transforms.image.resize import resize
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils import non_maximum_suppression


def _mask_aggregation(
        bbox, seg_prob, seg_weight,
        size, binary_thresh
):
    assert bbox.shape[0] == len(seg_prob)
    assert bbox.shape[0] == seg_weight.shape[0]

    aggregated_msk = np.zeros(size, dtype=np.float32)
    for bb, seg_pb, seg_w in zip(bbox, seg_prob, seg_weight):
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        if y_max - y_min > 0 and x_max - x_min > 0:
            seg_pb = resize(
                seg_pb.astype(np.float32)[None],
                (y_max - y_min, x_max - x_min))
            seg_m = (seg_pb >= binary_thresh).astype(np.float32)[0]
            aggregated_msk[y_min:y_max, x_min:x_max] += seg_m * seg_w

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
        seg_prob, bbox, cls_prob, size,
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
    * :math:`R` is the total number of RoIs produced in one image.
    * :math:`L` is the number of classes excluding the background.
    * :math:`RH` is the height of pooled image.
    * :math:`RW` is the height of pooled image.

    Args:
        seg_prob (array): A mask probability array whose shape is
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
        * **v_seg_prob**: Merged mask probability. Its shapes is \
            :math:`(N, RH, RW)`.
        * **v_bbox**: Bounding boxes for the merged masks. Its shape is \
            :math:`(N, 4)`.
        * **v_label**: Class labels for the merged masks. Its shape is \
            :math:`(N, )`.
        * **v_score**: Class probabilities for the merged masks. Its shape \
            is :math:`(N, )`.

    """

    seg_size = seg_prob.shape[1:]
    n_class = cls_prob.shape[1]

    v_seg_prob = []
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

        v_seg_prob_l = []
        v_bbox_l = []
        v_score_l = []

        for i, bb in enumerate(bbox_l):
            iou = bbox_iou(bbox, bb[np.newaxis, :])
            keep_indices = np.where(iou >= mask_merge_thresh)[0]
            seg_weight = cls_prob[keep_indices, label]
            seg_weight = seg_weight / seg_weight.sum()
            seg_prob_i = seg_prob[keep_indices]
            bbox_i = bbox[keep_indices]
            m_seg, m_bbox = _mask_aggregation(
                bbox_i, seg_prob_i, seg_weight, size, binary_thresh)
            if m_seg is not None and m_bbox is not None:
                m_seg = resize(m_seg, seg_size)
                m_seg = np.clip(m_seg, 0.0, 1.0)
                v_seg_prob_l.append(m_seg)
                v_bbox_l.append(m_bbox)
                v_score_l.append(score_l[i])

        if len(v_seg_prob_l) > 0:
            v_label_l = np.repeat(
                label - 1, len(v_score_l)).astype(np.int32)

            v_seg_prob += v_seg_prob_l
            v_bbox += v_bbox_l
            v_label.append(v_label_l)
            v_cls_prob.append(v_score_l)

    if len(v_seg_prob) > 0:
        v_seg_prob = np.concatenate(v_seg_prob)
        v_bbox = np.concatenate(v_bbox)
        v_label = np.concatenate(v_label)
        v_cls_prob = np.concatenate(v_cls_prob)
    else:
        v_seg_prob = np.empty((0, seg_size[0], seg_size[1]))
        v_bbox = np.empty((0, 4))
        v_label = np.empty((0, ))
        v_cls_prob = np.empty((0, ))
    return v_seg_prob, v_bbox, v_label, v_cls_prob
