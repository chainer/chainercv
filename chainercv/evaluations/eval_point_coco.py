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


def eval_point_coco(pred_points, pred_labels, pred_scores,
                    gt_points, gt_is_valids, gt_bboxes, gt_labels,
                    gt_areas, gt_crowdeds=None):
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
    gt_is_valids = iter(gt_is_valids)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)

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
    for i, (pred_point, pred_label, pred_score, gt_point, gt_is_valid,
            gt_bbox, gt_label,
            gt_area, gt_crowded) in enumerate(six.moves.zip(
                pred_points, pred_labels, pred_scores,
                gt_points, gt_is_valids, gt_bboxes, gt_labels,
                gt_areas, gt_crowdeds)):
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
            is_v = np.ones(len(pred_pnt))
            pred_annos.append(
                _create_anno(pred_pnt, is_v, None,
                             pred_lb, pred_sc,
                             img_id=img_id, anno_id=len(pred_annos) + 1,
                             ar=None, crw=0))
            existent_labels[pred_lb] = True

        for gt_pnt, gt_is_v, gt_bb, gt_lb, gt_ar, gt_crw in zip(
                gt_point, gt_is_valid, gt_bbox, gt_label, gt_area, gt_crowded):
            gt_annos.append(
                _create_anno(gt_pnt, gt_is_v, gt_bb, gt_lb, None,
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


def _create_anno(pnt, is_v, bb, lb, sc, img_id, anno_id, ar=None, crw=None):
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
    pnt = np.concatenate((pnt[:, [1, 0]], is_v[:, None]), axis=1)
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
