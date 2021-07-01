import itertools
import numpy as np
import os
import six

from chainercv.evaluations.eval_detection_coco import _redirect_stdout
from chainercv.evaluations.eval_detection_coco import _summarize

try:
    import pycocotools.coco
    import pycocotools.cocoeval
    import pycocotools.mask as mask_tools
    _available = True
except ImportError:
    _available = False


def eval_instance_segmentation_coco(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, gt_areas=None, gt_crowdeds=None):
    """Evaluate instance segmentations based on evaluation code of MS COCO.

    This function evaluates predicted instance segmentations obtained from
    a dataset by using average precision for each class.
    The code is based on the evaluation code used in MS COCO.

    Args:
        pred_masks (iterable of numpy.ndarray): See the table below.
        pred_labels (iterable of numpy.ndarray): See the table below.
        pred_scores (iterable of numpy.ndarray): See the table below.
        gt_masks (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
        gt_areas (iterable of numpy.ndarray): See the table below. If
            :obj:`None`, some scores are not returned.
        gt_crowdeds (iterable of numpy.ndarray): See the table below.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`pred_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`pred_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`pred_scores`, ":math:`[(R,)]`", :obj:`float32`, \
        --
        :obj:`gt_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`gt_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`gt_areas`, ":math:`[(R,)]`", \
        :obj:`float32`, --
        :obj:`gt_crowdeds`, ":math:`[(R,)]`", :obj:`bool`, --

    All inputs should have the same length. For more detailed explanation
    of the inputs, please refer to
    :class:`chainercv.datasets.COCOInstanceSegmentationDataset`.

    .. seealso::
        :class:`chainercv.datasets.COCOInstanceSegmentationDataset`.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below. The APs and ARs calculated with different iou
        thresholds, sizes of objects, and numbers of detections
        per image. For more details on the 12 patterns of evaluation metrics,
        please refer to COCO's official `evaluation page`_.

        .. csv-table::
            :header: key, type, description

            ap/iou=0.50:0.95/area=all/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_1]_
            ap/iou=0.50/area=all/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_1]_
            ap/iou=0.75/area=all/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_1]_
            ap/iou=0.50:0.95/area=small/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_1]_ [#coco_ins_eval_5]_
            ap/iou=0.50:0.95/area=medium/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_1]_ [#coco_ins_eval_5]_
            ap/iou=0.50:0.95/area=large/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_1]_ [#coco_ins_eval_5]_
            ar/iou=0.50:0.95/area=all/max_dets=1, *numpy.ndarray*, \
                [#coco_ins_eval_2]_
            ar/iou=0.50/area=all/max_dets=10, *numpy.ndarray*, \
                [#coco_ins_eval_2]_
            ar/iou=0.75/area=all/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_2]_
            ar/iou=0.50:0.95/area=small/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_2]_ [#coco_ins_eval_5]_
            ar/iou=0.50:0.95/area=medium/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_2]_ [#coco_ins_eval_5]_
            ar/iou=0.50:0.95/area=large/max_dets=100, *numpy.ndarray*, \
                [#coco_ins_eval_2]_ [#coco_ins_eval_5]_
            map/iou=0.50:0.95/area=all/max_dets=100, *float*, \
                [#coco_ins_eval_3]_
            map/iou=0.50/area=all/max_dets=100, *float*, \
                [#coco_ins_eval_3]_
            map/iou=0.75/area=all/max_dets=100, *float*, \
                [#coco_ins_eval_3]_
            map/iou=0.50:0.95/area=small/max_dets=100, *float*, \
                [#coco_ins_eval_3]_ [#coco_ins_eval_5]_
            map/iou=0.50:0.95/area=medium/max_dets=100, *float*, \
                [#coco_ins_eval_3]_ [#coco_ins_eval_5]_
            map/iou=0.50:0.95/area=large/max_dets=100, *float*, \
                [#coco_ins_eval_3]_ [#coco_ins_eval_5]_
            mar/iou=0.50:0.95/area=all/max_dets=1, *float*, \
                [#coco_ins_eval_4]_
            mar/iou=0.50/area=all/max_dets=10, *float*, \
                [#coco_ins_eval_4]_
            mar/iou=0.75/area=all/max_dets=100, *float*, \
                [#coco_ins_eval_4]_
            mar/iou=0.50:0.95/area=small/max_dets=100, *float*, \
                [#coco_ins_eval_4]_ [#coco_ins_eval_5]_
            mar/iou=0.50:0.95/area=medium/max_dets=100, *float*, \
                [#coco_ins_eval_4]_ [#coco_ins_eval_5]_
            mar/iou=0.50:0.95/area=large/max_dets=100, *float*, \
                [#coco_ins_eval_4]_ [#coco_ins_eval_5]_
            coco_eval, *pycocotools.cocoeval.COCOeval*, \
                result from :obj:`pycocotools`
            existent_labels, *numpy.ndarray*, \
                used labels \

    .. [#coco_ins_eval_1] An array of average precisions. \
        The :math:`l`-th value corresponds to the average precision \
        for class :math:`l`. If class :math:`l` does not exist in \
        either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
        value is set to :obj:`numpy.nan`.
    .. [#coco_ins_eval_2] An array of average recalls. \
        The :math:`l`-th value corresponds to the average precision \
        for class :math:`l`. If class :math:`l` does not exist in \
        either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
        value is set to :obj:`numpy.nan`.
    .. [#coco_ins_eval_3] The average of average precisions over classes.
    .. [#coco_ins_eval_4] The average of average recalls over classes.
    .. [#coco_ins_eval_5] Skip if :obj:`gt_areas` is :obj:`None`.

    """

    if not _available:
        raise ValueError(
            'Please install pycocotools \n'
            'pip install -e \'git+https://github.com/cocodataset/coco.git'
            '#egg=pycocotools&subdirectory=PythonAPI\'')

    gt_coco = pycocotools.coco.COCO()
    pred_coco = pycocotools.coco.COCO()

    pred_masks = iter(pred_masks)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_masks = iter(gt_masks)
    gt_labels = iter(gt_labels)
    if gt_areas is None:
        compute_area_dependent_metrics = False
        gt_areas = itertools.repeat(None)
    else:
        compute_area_dependent_metrics = True
        gt_areas = iter(gt_areas)
    gt_crowdeds = (iter(gt_crowdeds) if gt_crowdeds is not None
                   else itertools.repeat(None))

    images = []
    pred_annos = []
    gt_annos = []
    existent_labels = {}
    for i, (pred_mask, pred_label, pred_score, gt_mask, gt_label,
            gt_area, gt_crowded) in enumerate(six.moves.zip(
                pred_masks, pred_labels, pred_scores,
                gt_masks, gt_labels, gt_areas, gt_crowdeds)):
        size = pred_mask.shape[1:]
        if gt_area is None:
            gt_area = itertools.repeat(None)
        if gt_crowded is None:
            gt_crowded = itertools.repeat(None)
        # Starting ids from 1 is important when using COCO.
        img_id = i + 1

        for pred_msk, pred_lb, pred_sc in zip(
                pred_mask, pred_label, pred_score):
            pred_annos.append(
                _create_anno(pred_msk, pred_lb, pred_sc,
                             img_id=img_id, anno_id=len(pred_annos) + 1,
                             crw=0, ar=None))
            existent_labels[pred_lb] = True
        for gt_msk, gt_lb, gt_ar, gt_crw in zip(
                gt_mask, gt_label, gt_area, gt_crowded):
            gt_annos.append(
                _create_anno(gt_msk, gt_lb, None,
                             img_id=img_id, anno_id=len(gt_annos) + 1,
                             ar=gt_ar, crw=gt_crw))
            existent_labels[gt_lb] = True
        images.append({'id': img_id, 'height': size[0], 'width': size[1]})
    existent_labels = sorted(existent_labels.keys())

    pred_coco.dataset['categories'] = [{'id': i} for i in existent_labels]
    gt_coco.dataset['categories'] = [{'id': i} for i in existent_labels]
    pred_coco.dataset['annotations'] = pred_annos
    gt_coco.dataset['annotations'] = gt_annos
    pred_coco.dataset['images'] = images
    gt_coco.dataset['images'] = images

    with _redirect_stdout(open(os.devnull, 'w')):
        pred_coco.createIndex()
        gt_coco.createIndex()
        coco_eval = pycocotools.cocoeval.COCOeval(gt_coco, pred_coco, 'segm')
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
        'ap/iou=0.50:0.95/area=all/max_dets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.50/area=all/max_dets=100': {
            'ap': True, 'iou_thresh': 0.5, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.75/area=all/max_dets=100': {
            'ap': True, 'iou_thresh': 0.75, 'area_range': 'all',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=all/max_dets=1': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 1},
        'ar/iou=0.50:0.95/area=all/max_dets=10': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 10},
        'ar/iou=0.50:0.95/area=all/max_dets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
    }
    if compute_area_dependent_metrics:
        all_kwargs.update({
            'ap/iou=0.50:0.95/area=small/max_dets=100': {
                'ap': True, 'iou_thresh': None, 'area_range': 'small',
                'max_detection': 100},
            'ap/iou=0.50:0.95/area=medium/max_dets=100': {
                'ap': True, 'iou_thresh': None, 'area_range': 'medium',
                'max_detection': 100},
            'ap/iou=0.50:0.95/area=large/max_dets=100': {
                'ap': True, 'iou_thresh': None, 'area_range': 'large',
                'max_detection': 100},
            'ar/iou=0.50:0.95/area=small/max_dets=100': {
                'ap': False, 'iou_thresh': None, 'area_range': 'small',
                'max_detection': 100},
            'ar/iou=0.50:0.95/area=medium/max_dets=100': {
                'ap': False, 'iou_thresh': None, 'area_range': 'medium',
                'max_detection': 100},
            'ar/iou=0.50:0.95/area=large/max_dets=100': {
                'ap': False, 'iou_thresh': None, 'area_range': 'large',
                'max_detection': 100},
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


def _create_anno(msk, lb, sc, img_id, anno_id, ar=None, crw=None):
    H, W = msk.shape
    if crw is None:
        crw = False
    msk = np.asfortranarray(msk.astype(np.uint8))
    rle = mask_tools.encode(msk)
    if ar is None:
        # We compute dummy area to pass to pycocotools.
        # Note that area dependent scores are ignored afterwards.
        ar = mask_tools.area(rle)
    if crw is None:
        crw = False
    # Rounding is done to make the result consistent with COCO.
    anno = {
        'image_id': img_id, 'category_id': lb,
        'segmentation': rle,
        'area': ar,
        'id': anno_id,
        'iscrowd': crw}
    if sc is not None:
        anno.update({'score': sc})
    return anno
