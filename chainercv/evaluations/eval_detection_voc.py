from __future__ import division

import numpy as np
import six

from chainercv.utils.bbox.bbox_iou import bbox_iou


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate detection metrics based on evaluation code of PASCAL VOC.

    This function evaluates recall, precison and average precision with
    respect to a class as well as mean average precision.
    This evaluates predicted bounding boxes obtained from a dataset which
    has :math:`N` images.

    Mean average precision is calculated by taking a mean of average
    precisions for all classes which have at least one bounding box
    assigned by the predictions or the ground truth labels.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (list of numpy.ndarray): A list of :math:`N` bounding
            boxes. Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to :obj:`x_min, y_min, x_max, y_max`
            of a box.
        pred_labels (list of numpy.ndarray): A list of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (list of numpy.ndarray): A list of confidence scores for
            predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (list of numpy.ndarray): List of ground truth bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (list of numpy.ndarray): List of ground truth labels which
            are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (list of numpy.ndarray): List of boolean arrays which
            is organized similarly to :obj:`gt_bboxes`. This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        dict:

        This function returns a dictionary whose contents are listed
        below with key, value-type and the description of the value.

        * **map** (*float*): Mean Average Prediction.
        * **i (an integer corresponding to class id)** (*dict*): This is a \
            dictionary whose keys are :obj:`precision, recall, ap`, which \
            map to precision, recall and average precision with respect \
            to the class id **i**.

    """
    if not (len(pred_bboxes) == len(pred_labels) == len(pred_scores)
            == len(gt_bboxes) == len(gt_labels)):
        raise ValueError('Length of list inputs need to be same')
    n_img = len(pred_bboxes)

    valid_label = np.union1d(
        np.unique(np.concatenate(pred_labels)),
        np.unique(np.concatenate(gt_labels))).astype(np.int32)

    # Initial values stored in the dictionaries.
    empty_bbox = np.zeros((0, 4), dtype=np.float32)
    empty_label = np.zeros((0,), dtype=np.bool)

    # Organize predictions into Dict[l, List[bbox]]
    pred_bboxes_list = {l: [empty_bbox for _ in six.moves.range(n_img)]
                        for l in valid_label}
    pred_scores_list = {l: [empty_label for _ in six.moves.range(n_img)]
                        for l in valid_label}
    for n in six.moves.range(n_img):
        for l in valid_label:
            bboxes_l = []
            scores_l = []
            for r in six.moves.range(pred_bboxes[n].shape[0]):
                if l == pred_labels[n][r]:
                    bboxes_l.append(pred_bboxes[n][r])
                    scores_l.append(pred_scores[n][r])
            if len(bboxes_l) > 0:
                pred_bboxes_list[l][n] = np.stack(bboxes_l)
                pred_scores_list[l][n] = np.stack(scores_l)

    # Organize ground truths into Dict[l, List[bbox]]
    gt_bboxes_list = {l: [empty_bbox for _ in six.moves.range(n_img)]
                      for l in valid_label}
    gt_difficults_list = {l: [empty_label for _ in six.moves.range(n_img)]
                          for l in valid_label}
    for n in six.moves.range(n_img):
        for l in valid_label:
            gt_bboxes_l = []
            gt_difficults_l = []
            for r in six.moves.range(gt_bboxes[n].shape[0]):
                if l == gt_labels[n][r]:
                    gt_bboxes_l.append(gt_bboxes[n][r])
                    if gt_difficults is not None:
                        gt_difficults_l.append(gt_difficults[n][r])
                    else:
                        gt_difficults_l.append(
                            np.array(False, dtype=np.bool))
            if len(gt_bboxes_l) > 0:
                gt_bboxes_list[l][n] = np.stack(gt_bboxes_l)
                gt_difficults_list[l][n] = np.stack(gt_difficults_l)

    # Accumulate recacall, precison and ap
    results = {}
    for l in valid_label:
        rec, prec = _pred_and_rec_cls(
            pred_bboxes_list[l],
            pred_scores_list[l],
            gt_bboxes_list[l],
            gt_difficults_list[l],
            iou_thresh)
        ap = _voc_ap(rec, prec, use_07_metric=use_07_metric)
        results[l] = {}
        results[l]['recall'] = rec
        results[l]['precision'] = prec
        results[l]['ap'] = ap
    results['map'] = np.asscalar(np.mean(
        [results[l]['ap'] for l in valid_label]))
    return results


def _pred_and_rec_cls(
        bboxes, scores, gt_bboxes, gt_difficults, iou_thresh=0.5):
    # Calculate detection metrics with respect to a class.
    # This function is called only when there is at least one
    # prediction or ground truth box which is labeled as the class.
    # bboxes: List[numpy.ndarray]
    # scores: List[numpy.ndarray]
    # gt_bboxes: List[numpy.ndarray]
    # gt_difficults: List[numpy.ndarray]

    n_pos = 0  # The number of non difficult objects.
    selec = [None for _ in six.moves.range(len(gt_bboxes))]
    for n in six.moves.range(len(gt_bboxes)):
        n_gt_bbox = len(gt_bboxes[n])
        selec[n] = np.zeros(n_gt_bbox, dtype=np.bool)
        n_pos += np.sum(np.logical_not(gt_difficults[n]))

    # Make list of arrays into one array.
    # Example:
    # bboxes = [[bbox00, bbox01], [bbox10]]
    # bbox = array([bbox00, bbox01, bbox10])
    # index = [0, 0, 1]
    index = []
    for n in six.moves.range(len(scores)):
        for r in six.moves.range(len(scores[n])):
            index.append(n)
    index = np.array(index, dtype=np.int)
    conf = np.concatenate(scores)
    bbox = np.concatenate(bboxes)

    if n_pos == 0 or len(conf) == 0:
        return np.zeros((len(conf),)), np.zeros((len(conf),))

    # Reorder arrays by scores in descending order.
    si = np.argsort(-conf)
    index = index[si]
    bbox = bbox[si]

    nd = len(index)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in six.moves.range(nd):
        idx = index[d]
        bb = bbox[d]
        ioumax = -np.inf
        gt_bb = gt_bboxes[idx]

        if gt_bb.size > 0:
            # VOC evaluation follows integer typed bounding boxes.
            gt_bb_int = np.concatenate((gt_bb[:, :2], gt_bb[:, 2:] + 1),
                                       axis=1)
            bb_int = np.concatenate((bb[None][:, :2], bb[None][:, 2:] + 1),
                                    axis=1)
            iou = bbox_iou(gt_bb_int, bb_int)[:, 0]
            ioumax = np.max(iou)
            jmax = np.argmax(iou)

        if ioumax > iou_thresh:
            if not gt_difficults[idx][jmax]:
                if not selec[idx][jmax]:
                    tp[d] = 1
                    # assign detections to ground truth objects
                    selec[idx][jmax] = 1
                else:
                    fp[d] = 1
        else:
            fp[d] = 1

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(n_pos)
    prec = tp / np.maximum(fp + tp, np.finfo(np.float64).eps)
    return rec, prec


def _voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in six.moves.range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
