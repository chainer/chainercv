import itertools
import numpy as np
import os
import six

from chainercv.evaluations.eval_detection_coco import _redirect_stdout
from chainercv.evaluations.eval_detection_coco import _summarize

try:
    import pycocotools.coco
    import pycocotools.cocoeval
    _available = True
except ImportError:
    _available = False


def eval_keypoint_detection_coco(
        pred_points, pred_labels, pred_scores,
        gt_points, gt_visibles, gt_labels=None, gt_bboxes=None,
        gt_areas=None, gt_crowdeds=None):
    """Evaluate keypoint detection based on evaluation code of MS COCO.

    This function evaluates predicted keypints obtained by using average
    precision for each class.
    The code is based on the evaluation code used in MS COCO.

    Args:
        pred_points (iterable of numpy.ndarray): See the table below.
        pred_labels (iterable of numpy.ndarray): See the table below.
        pred_scores (iterable of numpy.ndarray): See the table below.
            This is used to rank instances. Note that this is not
            the confidene for each keypoint.
        gt_points (iterable of numpy.ndarray): See the table below.
        gt_visibles (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
        gt_bboxes (iterable of numpy.ndarray): See the table below.
            This is optional. If this is :obj:`None`, the ground truth
            bounding boxes are esitmated from the ground truth
            keypoints.
        gt_areas (iterable of numpy.ndarray): See the table below. If
            :obj:`None`, some scores are not returned.
        gt_crowdeds (iterable of numpy.ndarray): See the table below.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`pred_points`, ":math:`[(R, K, 2)]`", :obj:`float32`, \
        ":math:`(y, x)`"
        :obj:`pred_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`pred_scores`, ":math:`[(R,)]`", :obj:`float32`, \
        --
        :obj:`gt_points`, ":math:`[(R, K, 2)]`", :obj:`float32`, \
        ":math:`(y, x)`"
        :obj:`gt_visibles`, ":math:`[(R, K)]`", :obj:`bool`, --
        :obj:`gt_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`gt_bboxes`, ":math:`[(R, 4)]`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`gt_areas`, ":math:`[(R,)]`", \
        :obj:`float32`, --
        :obj:`gt_crowdeds`, ":math:`[(R,)]`", :obj:`bool`, --


    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below. The APs and ARs calculated with different iou
        thresholds, sizes of objects, and numbers of detections
        per image. For more details on the 12 patterns of evaluation metrics,
        please refer to COCO's official `evaluation page`_.

        .. csv-table::
            :header: key, type, description

            ap/iou=0.50:0.95/area=all/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_1]_
            ap/iou=0.50/area=all/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_1]_
            ap/iou=0.75/area=all/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_1]_
            ap/iou=0.50:0.95/area=medium/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_1]_ [#coco_kp_eval_5]_
            ap/iou=0.50:0.95/area=large/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_1]_ [#coco_kp_eval_5]_
            ar/iou=0.50:0.95/area=all/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_2]_
            ar/iou=0.50/area=all/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_2]_
            ar/iou=0.75/area=all/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_2]_
            ar/iou=0.50:0.95/area=medium/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_2]_ [#coco_kp_eval_5]_
            ar/iou=0.50:0.95/area=large/max_dets=20, *numpy.ndarray*, \
                [#coco_kp_eval_2]_ [#coco_kp_eval_5]_
            map/iou=0.50:0.95/area=all/max_dets=20, *float*, \
                [#coco_kp_eval_3]_
            map/iou=0.50/area=all/max_dets=20, *float*, \
                [#coco_kp_eval_3]_
            map/iou=0.75/area=all/max_dets=20, *float*, \
                [#coco_kp_eval_3]_
            map/iou=0.50:0.95/area=medium/max_dets=20, *float*, \
                [#coco_kp_eval_3]_ [#coco_kp_eval_5]_
            map/iou=0.50:0.95/area=large/max_dets=20, *float*, \
                [#coco_kp_eval_3]_ [#coco_kp_eval_5]_
            mar/iou=0.50:0.95/area=all/max_dets=20, *float*, \
                [#coco_kp_eval_4]_
            mar/iou=0.50/area=all/max_dets=20, *float*, \
                [#coco_kp_eval_4]_
            mar/iou=0.75/area=all/max_dets=20, *float*, \
                [#coco_kp_eval_4]_
            mar/iou=0.50:0.95/area=medium/max_dets=20, *float*, \
                [#coco_kp_eval_4]_ [#coco_kp_eval_5]_
            mar/iou=0.50:0.95/area=large/max_dets=20, *float*, \
                [#coco_kp_eval_4]_ [#coco_kp_eval_5]_
            coco_eval, *pycocotools.cocoeval.COCOeval*, \
                result from :obj:`pycocotools`
            existent_labels, *numpy.ndarray*, \
                used labels \

    .. [#coco_kp_eval_1] An array of average precisions. \
        The :math:`l`-th value corresponds to the average precision \
        for class :math:`l`. If class :math:`l` does not exist in \
        either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
        value is set to :obj:`numpy.nan`.
    .. [#coco_kp_eval_2] An array of average recalls. \
        The :math:`l`-th value corresponds to the average precision \
        for class :math:`l`. If class :math:`l` does not exist in \
        either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
        value is set to :obj:`numpy.nan`.
    .. [#coco_kp_eval_3] The average of average precisions over classes.
    .. [#coco_kp_eval_4] The average of average recalls over classes.
    .. [#coco_kp_eval_5] Skip if :obj:`gt_areas` is :obj:`None`.

    """
    if not _available:
        raise ValueError(
            'Please install pycocotools \n'
            'pip install -e \'git+https://github.com/cocodataset/coco.git'
            '#egg=pycocotools&subdirectory=PythonAPI\'')

    gt_coco = pycocotools.coco.COCO()
    pred_coco = pycocotools.coco.COCO()

    pred_points = iter(pred_points)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_points = iter(gt_points)
    gt_visibles = iter(gt_visibles)
    gt_labels = iter(gt_labels)
    gt_bboxes = (iter(gt_bboxes) if gt_bboxes is not None
                 else itertools.repeat(None))
    if gt_areas is None:
        compute_area_dependent_metrics = False
        gt_areas = itertools.repeat(None)
    else:
        compute_area_dependent_metrics = True
        gt_areas = iter(gt_areas)
    gt_crowdeds = (iter(gt_crowdeds) if gt_crowdeds is not None
                   else itertools.repeat(None))

    ids = []
    pred_annos = []
    gt_annos = []
    existent_labels = {}
    for i, (pred_point, pred_label, pred_score, gt_point, gt_visible,
            gt_label, gt_bbox,
            gt_area, gt_crowded) in enumerate(six.moves.zip(
                pred_points, pred_labels, pred_scores,
                gt_points, gt_visibles, gt_labels, gt_bboxes,
                gt_areas, gt_crowdeds)):
        if gt_bbox is None:
            gt_bbox = itertools.repeat(None)
        if gt_area is None:
            gt_area = itertools.repeat(None)
        if gt_crowded is None:
            gt_crowded = itertools.repeat(None)
        # Starting ids from 1 is important when using COCO.
        img_id = i + 1

        for pred_pnt, pred_lb, pred_sc in zip(pred_point, pred_label,
                                              pred_score):
            # http://cocodataset.org/#format-results
            # Visibility flag is currently not used for evaluation
            v = np.ones(len(pred_pnt))
            pred_annos.append(
                _create_anno(pred_pnt, v,
                             pred_lb, pred_sc, None,
                             img_id=img_id, anno_id=len(pred_annos) + 1,
                             ar=None, crw=0))
            existent_labels[pred_lb] = True

        for gt_pnt, gt_v, gt_lb, gt_bb, gt_ar, gt_crw in zip(
                gt_point, gt_visible, gt_label, gt_bbox, gt_area, gt_crowded):
            gt_annos.append(
                _create_anno(gt_pnt, gt_v, gt_lb, None, gt_bb,
                             img_id=img_id, anno_id=len(gt_annos) + 1,
                             ar=gt_ar, crw=gt_crw))
        ids.append({'id': img_id})
    existent_labels = sorted(existent_labels.keys())

    pred_coco.dataset['categories'] = [{'id': i} for i in existent_labels]
    gt_coco.dataset['categories'] = [{'id': i} for i in existent_labels]
    pred_coco.dataset['annotations'] = pred_annos
    gt_coco.dataset['annotations'] = gt_annos
    pred_coco.dataset['images'] = ids
    gt_coco.dataset['images'] = ids

    with _redirect_stdout(open(os.devnull, 'w')):
        pred_coco.createIndex()
        gt_coco.createIndex()
        coco_eval = pycocotools.cocoeval.COCOeval(
            gt_coco, pred_coco, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()

    results = {'coco_eval': coco_eval}
    p = coco_eval.params
    common_kwargs = {
        'prec': coco_eval.eval['precision'],
        'rec': coco_eval.eval['recall'],
        'iou_threshs': p.iouThrs,
        'area_ranges': p.areaRngLbl,
        'max_detection_list': p.maxDets,
    }
    all_kwargs = {
        'ap/iou=0.50:0.95/area=all/max_dets=20': {
            'ap': True, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 20},
        'ap/iou=0.50/area=all/max_dets=20': {
            'ap': True, 'iou_thresh': 0.5, 'area_range': 'all',
            'max_detection': 20},
        'ap/iou=0.75/area=all/max_dets=20': {
            'ap': True, 'iou_thresh': 0.75, 'area_range': 'all',
            'max_detection': 20},
        'ar/iou=0.50:0.95/area=all/max_dets=20': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 20},
        'ar/iou=0.50/area=all/max_dets=20': {
            'ap': False, 'iou_thresh': 0.5, 'area_range': 'all',
            'max_detection': 20},
        'ar/iou=0.75/area=all/max_dets=20': {
            'ap': False, 'iou_thresh': 0.75, 'area_range': 'all',
            'max_detection': 20},
    }
    if compute_area_dependent_metrics:
        all_kwargs.update({
            'ap/iou=0.50:0.95/area=medium/max_dets=20': {
                'ap': True, 'iou_thresh': None, 'area_range': 'medium',
                'max_detection': 20},
            'ap/iou=0.50:0.95/area=large/max_dets=20': {
                'ap': True, 'iou_thresh': None, 'area_range': 'large',
                'max_detection': 20},
            'ar/iou=0.50:0.95/area=medium/max_dets=20': {
                'ap': False, 'iou_thresh': None, 'area_range': 'medium',
                'max_detection': 20},
            'ar/iou=0.50:0.95/area=large/max_dets=20': {
                'ap': False, 'iou_thresh': None, 'area_range': 'large',
                'max_detection': 20},
        })

    for key, kwargs in all_kwargs.items():
        kwargs.update(common_kwargs)
        metrics, mean_metric = _summarize(**kwargs)

        # pycocotools ignores classes that are not included in
        # either gt or prediction, but lies between 0 and
        # the maximum label id.
        # We set values for these classes to np.nan.
        results[key] = np.nan * np.ones(np.max(existent_labels) + 1)
        results[key][existent_labels] = metrics
        results['m' + key] = mean_metric

    results['existent_labels'] = existent_labels
    return results


def _create_anno(pnt, v, lb, sc, bb, img_id, anno_id, ar=None, crw=None):
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L342
    y_min = np.min(pnt[:, 0])
    x_min = np.min(pnt[:, 1])
    y_max = np.max(pnt[:, 0])
    x_max = np.max(pnt[:, 1])
    if ar is None:
        ar = (y_max - y_min) * (x_max - x_min)

    if crw is None:
        crw = False
    # Rounding is done to make the result consistent with COCO.

    if bb is None:
        bb_xywh = [x_min, y_min, x_max - x_min, y_max - y_min]
    else:
        bb_xywh = [bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0]]
    pnt = np.concatenate((pnt[:, [1, 0]], v[:, None]), axis=1)
    anno = {
        'image_id': img_id, 'category_id': lb,
        'keypoints': pnt.reshape((-1)).tolist(),
        'area': ar,
        'bbox': bb_xywh,
        'id': anno_id,
        'iscrowd': crw,
        'num_keypoints': (pnt[:, 0] > 0).sum()
    }
    if sc is not None:
        anno.update({'score': sc})
    return anno
