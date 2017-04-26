import numpy as np

import chainer


def eval_detection(
        bboxes, labels, confs, gt_bboxes, gt_labels, n_class,
        minoverlap=0.5, use_07_metric=False):
    """Calculate deterction metrics.

    This function evaluates recall, precison and average precision with
    respect to a class as well as mean average precision.
    This evaluates predicted bounding boxes obtained from a dataset which
    has :math:`N` images.

    Args:
        bboxes (list of numpy.ndarray): A list of bounding boxes.
            The index to this list corresponds to the index of the data
            obtained from the base dataset. Length of the list is :math:`N`.
            The element of :obj:`bboxes` is coordinates of bounding
            boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to :obj:`x_min, y_min, x_max, y_max`
            of a box.
        labels (list of numpy.ndarray): A list of labels.
            Similar to :obj:`bboxes`, its index corresponds to an
            index for the base dataset. Its length :math:`N` list.
        confs (list of numpy.ndarray): A list of confidence scores for
            predicted bounding boxes. Similar to :obj:`bboxes`,
            its index corresponds to an index for the base dataset.
            Its length :math:`N` list.
        gt_bboxes (list of numpy.ndarray): List of ground truth bounding boxes
            which are organized similarly to :obj:`bboxes_cls`.
        gt_labels (list of numpy.ndarray): List of ground truth labels which
            are organized similarly to :obj:`labels`.
        minoverlap (float): A prediction is correct if its intersection of
            union with the ground truth is above this value.
        use_07_metric (bool): Whether to use Pascal VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        dict:

        This function returns a dictionary whose contents are listed
        below with key, value-type and the description of the value.

        * **map** (*float*): mAP calculated from the prediction and the\
            ground truth.
        * **i (an integer corresponding to class id)** (*dict*): This is a \
            dictionary whose keys are :obj:`precision, recall, ap`, which \
            maps to precision, recall and average precision with respect \
            to the class id **i**.

    """
    assert len(bboxes) == len(gt_bboxes)
    
    n_img = len(bboxes)
    _bboxes = [[None for _ in xrange(n_img)]
                for _ in xrange(n_class)]
    _confs = [[None for _ in xrange(n_img)]
                for _ in xrange(n_class)]

    for i in range(n_img):
        for cls in range(n_class):
            bboxes_cls = []
            confs_cls = []
            for j in range(bboxes[i].shape[0]):
                if cls == labels[i][j]:
                    bboxes_cls.append(bboxes[i][j])
                    confs_cls.append(confs[i][j])
            _bboxes[cls][i] = np.stack(bboxes_cls)
            _confs[cls][i] = np.stack(confs_cls)

    _gt_bboxes = [[None for _ in xrange(n_img)]
                  for _ in xrange(n_class)]
    for i in range(n_img):
        for cls in range(n_class):
            gt_bboxes_cls = []
            for j in range(gt_bboxes[i].shape[0]):
                if cls == gt_labels[i][j]:
                    gt_bboxes_cls.append(gt_bboxes[i][j])
            _gt_bboxes[cls][i] = np.stack(gt_bboxes_cls)

    results = {}
    for cls in range(n_class):
        rec, prec, ap = _eval_detection_cls(
            _bboxes[cls], _confs[cls], _gt_bboxes[cls],
            minoverlap, use_07_metric)
        results[cls] = {}
        results[cls]['recall'] = rec
        results[cls]['precision'] = prec
        results[cls]['ap'] = ap
    
    results['map'] = np.asscalar(np.mean(
        [results[cls]['ap'] for cls in range(n_class)]))
    return results


def _eval_detection_cls(
        bboxes_cls, confs_cls, gt_bboxes_cls,
        minoverlap=0.5, use_07_metric=False):
    # Calculate deterction metrics with respect to a class.
    npos = 0
    gt_det_cls = [None for i in range(len(gt_bboxes_cls))]
    for i in range(len(gt_bboxes_cls)):
        n_gt_bbox = len(gt_bboxes_cls[i])
        gt_det_cls[i] = np.zeros(n_gt_bbox)
        npos += n_gt_bbox

    # load the detection result
    indices = []
    for i in range(len(confs_cls)):
        for j in range(len(confs_cls[i])):
            indices.append(i)
    indices = np.array(indices, dtype=np.int)
    conf = np.concatenate(confs_cls)
    bbox = np.concatenate(bboxes_cls)
    if len(conf) == 0:
        return None

    si = np.argsort(-conf)
    indices = indices[si]
    bbox = bbox[si]

    # assign detections to ground truth objects
    nd = len(conf)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        index = indices[d]
        bb = bbox[d]
        ovmax = 0
        
        for j in range(len(gt_bboxes_cls[index])):
            bbgt = gt_bboxes_cls[index][j]
            ov = _iou_ratio(bb, bbgt)
            if ov > ovmax:
                ovmax = ov
                jmax = j

        if ovmax >= minoverlap:
            if not gt_det_cls[index][jmax]:
                tp[d] = 1
                gt_det_cls[index][jmax] = 1
            else:
                fp[d] = 1
        else:
            fp[d] = 1

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / npos
    prec = tp / np.maximum(fp + tp, np.finfo(np.float64).eps)

    ap = _voc_ap(rec, prec, use_07_metric=use_07_metric)
    return rec, prec, ap


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
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _iou_ratio(bbox_1, bbox_2):
    # Compute IoU between bounding boxes.

    bi = [np.maximum(bbox_1[0], bbox_2[0]),
          np.maximum(bbox_1[1], bbox_2[1]),
          np.minimum(bbox_1[2], bbox_2[2]),
          np.minimum(bbox_1[3], bbox_2[3])]

    iw = np.maximum(bi[2] - bi[0] + 1, 0.)
    ih = np.maximum(bi[3] - bi[1] + 1, 0.)

    overlap = 0
    
    inter = iw * ih

    # union
    uni = ((bbox_1[2] - bbox_1[0] + 1.) * (bbox_1[3] - bbox_1[1] + 1.) +
           (bbox_2[2] - bbox_2[0] + 1.) * (bbox_2[3] - bbox_2[1] + 1.) - inter)

    if ih > 0 and iw > 0:
        overlap = inter / uni
    return overlap
