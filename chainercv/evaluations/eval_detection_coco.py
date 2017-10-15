import contextlib
import itertools
import numpy as np
import os
import six
import sys

try:
    import pycocotools.coco
    import pycocotools.cocoeval
    _availabel = True
except ImportError:
    _availabel = False


def eval_detection_coco(pred_bboxes, pred_labels, pred_scores, gt_bboxes,
                        gt_labels, gt_crowdeds=None, gt_areas=None):
    """Evaluate detections based on evaluation code of MS COCO.

    This function evaluates predicted bounding boxes obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in MS COCO.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to :obj:`y_min, x_min, y_max, x_max`
            of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.

    """
    if not _availabel:
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
        ev = pycocotools.cocoeval.COCOeval(gt_coco, pred_coco, 'bbox')
        ev.evaluate()
        ev.accumulate()

    results = {'coco_eval': ev}
    p = ev.params
    common_kwargs = {
        'prec': ev.eval['precision'],
        'rec': ev.eval['recall'],
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
