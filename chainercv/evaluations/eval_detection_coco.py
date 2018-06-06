import contextlib
import itertools
import numpy as np
import os
import six
import sys

try:
    import pycocotools.coco
    import pycocotools.cocoeval
    _available = True
except ImportError:
    _available = False


def eval_detection_coco(pred_bboxes, pred_labels, pred_scores, gt_bboxes,
                        gt_labels, gt_areas=None, gt_crowdeds=None):
    """Evaluate detections based on evaluation code of MS COCO.

    This function evaluates predicted bounding boxes obtained from a dataset
    by using average precision for each class.
    The code is based on the evaluation code used in MS COCO.

    .. _`evaluation page`: http://cocodataset.org/#detections-eval

    Args:
        pred_bboxes (iterable of numpy.ndarray): See the table below.
        pred_labels (iterable of numpy.ndarray): See the table below.
        pred_scores (iterable of numpy.ndarray): See the table below.
        gt_bboxes (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
        gt_areas (iterable of numpy.ndarray): See the table below.
        gt_crowdeds (iterable of numpy.ndarray): See the table below.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`pred_bboxes`, ":math:`[(R, 4)]`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`pred_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`pred_scores`, ":math:`[(R,)]`", :obj:`float32`, \
        --
        :obj:`gt_bboxes`, ":math:`[(R, 4)]`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`gt_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`gt_areas`, ":math:`[(R,)]`", \
        :obj:`float32`, --
        :obj:`gt_crowdeds`, ":math:`[(R,)]`", :obj:`bool`, --

    All inputs should have the same length. For more detailed explanation
    of the inputs, please refer to
    :class:`chainercv.datasets.COCOBboxDataset`.

    .. seealso::
        :class:`chainercv.datasets.COCOBboxDataset`.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below. Each key contains four information: AP or AR, the iou
        thresholds, the size of objects, and the number of detections
        per image. For more details on the 12 patterns of evaluation metrics,
        please refer to COCO's official `evaluation page`_.

        * **ap/iou=0.50:0.95/area=all/maxDets=100** (*numpy.ndarray*): An \
            array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **ap/iou=0.50/area=all/maxDets=100** (*numpy.ndarray*): See above.
        * **ap/iou=0.75/area=all/maxDets=100** (*numpy.ndarray*): See above.
        * **ap/iou=0.50:0.95/area=small/maxDets=100** (*numpy.ndarray*): See \
            above.
        * **ap/iou=0.50:0.95/area=medium/maxDets=100** (*numpy.ndarray*): See \
            above.
        * **ap/iou=0.50:0.95/area=large/maxDets=100** (*numpy.ndarray*): See \
            above.
        * **ar/iou=0.50:0.95/area=all/maxDets=1** (*numpy.array*): An \
            array of average recalls. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **ar/iou=0.50:0.95/area=all/maxDets=10** (*numpy.array*): See above.
        * **ar/iou=0.50:0.95/area=all/maxDets=100** (*numpy.array*): See above.
        * **ar/iou=0.50:0.95/area=small/maxDets=100** (*numpy.array*): See \
            above.
        * **ar/iou=0.50:0.95/area=medium/maxDets=100** (*numpy.array*): See \
            above.
        * **ar/iou=0.50:0.95/area=large/maxDets=100** (*numpy.array*): See \
            above.
        * **map/iou=0.50:0.95/area=all/maxDets=100** (*float*): The average \
            of Average Precisions over classes.
        * **map/iou=0.50/area=all/maxDets=100** (*float*): See above.
        * **map/iou=0.75/area=all/maxDets=100** (*float*): See above.
        * **map/iou=0.50:0.95/area=small/maxDets=100** (*float*): See \
            above.
        * **map/iou=0.50:0.95/area=medium/maxDets=100** (*float*): See above.
        * **map/iou=0.50:0.95/area=large/maxDets=100** (*float*): See above.
        * **mar/iou=0.50:0.95/area=all/maxDets=1** (*float*): The average \
            of average recalls over classes.
        * **mar/iou=0.50:0.95/area=all/maxDets=10** (*float*): See above.
        * **mar/iou=0.50:0.95/area=all/maxDets=100** (*float*): See above.
        * **mar/iou=0.50:0.95/area=small/maxDets=100** (*float*): See above.
        * **mar/iou=0.50:0.95/area=medium/maxDets=100** (*float*): See above.
        * **mar/iou=0.50:0.95/area=large/maxDets=100** (*float*): See above.
        * **coco_eval** (*pycocotools.cocoeval.COCOeval*): The \
            :class:`pycocotools.cocoeval.COCOeval` object used to conduct \
            evaluation.

    """
    if not _available:
        raise ValueError(
            'Please install pycocotools \n'
            'pip install -e \'git+https://github.com/pdollar/coco.git'
            '#egg=pycocotools&subdirectory=PythonAPI\'')

    gt_coco = pycocotools.coco.COCO()
    pred_coco = pycocotools.coco.COCO()

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    gt_crowdeds = (iter(gt_crowdeds) if gt_crowdeds is not None
                   else itertools.repeat(None))
    gt_areas = (iter(gt_areas) if gt_areas is not None
                else itertools.repeat(None))

    images = list()
    pred_anns = list()
    gt_anns = list()
    unique_labels = dict()
    for i, (pred_bbox, pred_label, pred_score, gt_bbox, gt_label,
            gt_crowded, gt_area) in enumerate(six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_crowdeds, gt_areas)):
        if gt_crowded is None:
            gt_crowded = itertools.repeat(None)
        if gt_area is None:
            gt_area = itertools.repeat(None)
        # Starting ids from 1 is important when using COCO.
        img_id = i + 1

        for pred_bb, pred_lbl, pred_sc in zip(pred_bbox, pred_label,
                                              pred_score):
            pred_anns.append(
                _create_ann(pred_bb, pred_lbl, pred_sc,
                            img_id=img_id, ann_id=len(pred_anns) + 1,
                            crw=0, ar=None))
            unique_labels[pred_lbl] = True

        for gt_bb, gt_lbl, gt_crw, gt_ar in zip(
                gt_bbox, gt_label, gt_crowded, gt_area):
            gt_anns.append(
                _create_ann(gt_bb, gt_lbl, None,
                            img_id=img_id, ann_id=len(gt_anns) + 1,
                            crw=gt_crw, ar=gt_ar))
            unique_labels[gt_lbl] = True
        images.append({'id': img_id})

    pred_coco.dataset['categories'] = [{'id': i} for i in unique_labels.keys()]
    gt_coco.dataset['categories'] = [{'id': i} for i in unique_labels.keys()]
    pred_coco.dataset['annotations'] = pred_anns
    gt_coco.dataset['annotations'] = gt_anns
    pred_coco.dataset['images'] = images
    gt_coco.dataset['images'] = images

    with _redirect_stdout(open(os.devnull, 'w')):
        pred_coco.createIndex()
        gt_coco.createIndex()
        coco_eval = pycocotools.cocoeval.COCOeval(gt_coco, pred_coco, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()

    results = {'coco_eval': coco_eval}
    p = coco_eval.params
    common_kwargs = {
        'prec': coco_eval.eval['precision'],
        'rec': coco_eval.eval['recall'],
        'iou_threshs': p.iouThrs,
        'area_ranges': p.areaRngLbl,
        'max_detection_list': p.maxDets}
    all_kwargs = {
        'ap/iou=0.50:0.95/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.50/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': 0.5, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.75/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': 0.75, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=small/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'small',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=medium/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'medium',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=large/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'large',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=all/maxDets=1': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 1},
        'ar/iou=0.50:0.95/area=all/maxDets=10': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 10},
        'ar/iou=0.50:0.95/area=all/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=small/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'small',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=medium/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'medium',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=large/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'large',
            'max_detection': 100},
    }

    for key, kwargs in all_kwargs.items():
        kwargs.update(common_kwargs)
        metrics, mean_metric = _summarize(**kwargs)
        results[key] = metrics
        results['m' + key] = mean_metric
    return results


def _create_ann(bb, lbl, sc, img_id, ann_id, crw=None, ar=None):
    y_min = bb[0]
    x_min = bb[1]
    y_max = bb[2]
    x_max = bb[3]
    height = y_max - y_min
    width = x_max - x_min
    if crw is None:
        crw = False
    if ar is None:
        ar = height * width
    # Rounding is done to make the result consistent with COCO.
    ann = {
        'image_id': img_id, 'category_id': lbl,
        'bbox': [np.round(x_min, 2), np.round(y_min, 2),
                 np.round(width, 2), np.round(height, 2)],
        'segmentation': [x_min, y_min, x_min, y_max,
                         x_max, y_max, x_max, y_min],
        'area': ar,
        'id': ann_id,
        'iscrowd': crw}
    if sc is not None:
        ann.update({'score': sc})
    return ann


def _summarize(
        prec, rec, iou_threshs, area_ranges,
        max_detection_list,
        ap=True, iou_thresh=None, area_range='all',
        max_detection=100):
    a_idx = area_ranges.index(area_range)
    m_idx = max_detection_list.index(max_detection)
    if ap:
        s = prec.copy()  # (T, R, K, A, M)
        if iou_thresh is not None:
            s = s[iou_thresh == iou_threshs]
        s = s[:, :, :, a_idx, m_idx]
    else:
        s = rec.copy()  # (T, K, A, M)
        if iou_thresh is not None:
            s = s[iou_thresh == iou_threshs]
        s = s[:, :, a_idx, m_idx]

    s[s == -1] = np.nan
    s = s.reshape((-1, s.shape[-1]))
    valid_classes = np.any(np.logical_not(np.isnan(s)), axis=0)
    class_s = np.nan * np.ones(len(valid_classes), dtype=np.float32)
    class_s[valid_classes] = np.nanmean(s[:, valid_classes], axis=0)

    if not np.any(valid_classes):
        mean_s = np.nan
    else:
        mean_s = np.nanmean(class_s)
    return class_s, mean_s


@contextlib.contextmanager
def _redirect_stdout(target):
    original = sys.stdout
    sys.stdout = target
    yield
    sys.stdout = original
