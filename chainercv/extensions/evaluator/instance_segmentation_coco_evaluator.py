import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_instance_segmentation_coco
from chainercv.utils import apply_to_iterator

try:
    import pycocotools.coco  # NOQA
    _available = True
except ImportError:
    _available = False


class InstanceSegmentationCOCOEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a instance segmentation model by MS COCO metric.

    This extension iterates over an iterator and evaluates the prediction
    results.
    The results consist of average precisions (APs) and average
    recalls (ARs) as well as the mean of each (mean average precision and mean
    average recall).
    This extension reports the following values with keys.
    Please note that if
    :obj:`label_names` is not specified, only the mAPs and  mARs are reported.

    The underlying dataset of the iterator is assumed to return
    :obj:`img, mask, label` or :obj:`img, mask, label, area, crowded`.

    .. csv-table::
        :header: key, description

        ap/iou=0.50:0.95/area=all/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_1]_
        ap/iou=0.50/area=all/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_1]_
        ap/iou=0.75/area=all/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_1]_
        ap/iou=0.50:0.95/area=small/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_1]_ [#coco_ins_ext_5]_
        ap/iou=0.50:0.95/area=medium/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_1]_ [#coco_ins_ext_5]_
        ap/iou=0.50:0.95/area=large/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_1]_ [#coco_ins_ext_5]_
        ar/iou=0.50:0.95/area=all/max_dets=1/<label_names[l]>, \
            [#coco_ins_ext_2]_
        ar/iou=0.50/area=all/max_dets=10/<label_names[l]>, \
            [#coco_ins_ext_2]_
        ar/iou=0.75/area=all/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_2]_
        ar/iou=0.50:0.95/area=small/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_2]_ [#coco_ins_ext_5]_
        ar/iou=0.50:0.95/area=medium/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_2]_ [#coco_ins_ext_5]_
        ar/iou=0.50:0.95/area=large/max_dets=100/<label_names[l]>, \
            [#coco_ins_ext_2]_ [#coco_ins_ext_5]_
        map/iou=0.50:0.95/area=all/max_dets=100, \
            [#coco_ins_ext_3]_
        map/iou=0.50/area=all/max_dets=100, \
            [#coco_ins_ext_3]_
        map/iou=0.75/area=all/max_dets=100, \
            [#coco_ins_ext_3]_
        map/iou=0.50:0.95/area=small/max_dets=100, \
            [#coco_ins_ext_3]_ [#coco_ins_ext_5]_
        map/iou=0.50:0.95/area=medium/max_dets=100, \
            [#coco_ins_ext_3]_ [#coco_ins_ext_5]_
        map/iou=0.50:0.95/area=large/max_dets=100, \
            [#coco_ins_ext_3]_ [#coco_ins_ext_5]_
        ar/iou=0.50:0.95/area=all/max_dets=1, \
            [#coco_ins_ext_4]_
        ar/iou=0.50/area=all/max_dets=10, \
            [#coco_ins_ext_4]_
        ar/iou=0.75/area=all/max_dets=100, \
            [#coco_ins_ext_4]_
        ar/iou=0.50:0.95/area=small/max_dets=100, \
            [#coco_ins_ext_4]_ [#coco_ins_ext_5]_
        ar/iou=0.50:0.95/area=medium/max_dets=100, \
            [#coco_ins_ext_4]_ [#coco_ins_ext_5]_
        ar/iou=0.50:0.95/area=large/max_dets=100, \
            [#coco_ins_ext_4]_ [#coco_ins_ext_5]_

    .. [#coco_ins_ext_1] Average precision for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        If class :math:`l` does not exist in either :obj:`pred_labels` or \
        :obj:`gt_labels`, the corresponding value is set to :obj:`numpy.nan`.
    .. [#coco_ins_ext_2] Average recall for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        If class :math:`l` does not exist in either :obj:`pred_labels` or \
        :obj:`gt_labels`, the corresponding value is set to :obj:`numpy.nan`.
    .. [#coco_ins_ext_3] The average of average precisions over classes.
    .. [#coco_ins_ext_4] The average of average recalls over classes.
    .. [#coco_ins_ext_5] Skip if :obj:`gt_areas` is :obj:`None`.

    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, mask, label, area, crowded`.
        target (chainer.Link): A detection link. This link must have
            :meth:`predict` method that takes a list of images and returns
            :obj:`masks`, :obj:`labels` and :obj:`scores`.
        label_names (iterable of strings): An iterable of names of classes.
            If this value is specified, average precision and average
            recalls for each class are reported.

    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target,
            label_names=None):
        if not _available:
            raise ValueError(
                'Please install pycocotools \n'
                'pip install pycocotools')
        super(InstanceSegmentationCOCOEvaluator, self).__init__(
            iterator, target)
        self.label_names = label_names

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)
        # delete unused iterators explicitly
        del in_values

        pred_masks, pred_labels, pred_scores = out_values

        if len(rest_values) == 2:
            gt_masks, gt_labels = rest_values
            gt_areas = None
            gt_crowdeds = None
        elif len(rest_values) == 4:
            gt_masks, gt_labels, gt_areas, gt_crowdeds =\
                rest_values
        else:
            raise ValueError('the dataset should return '
                             'sets of (img, mask, label) or sets of '
                             '(img, mask, label, area, crowded).')

        result = eval_instance_segmentation_coco(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, gt_areas, gt_crowdeds)

        report = {}
        for key in result.keys():
            if key.startswith('map') or key.startswith('mar'):
                report[key] = result[key]

        if self.label_names is not None:
            for key in result.keys():
                if key.startswith('ap') or key.startswith('ar'):
                    for l, label_name in enumerate(self.label_names):
                        report_key = '{}/{:s}'.format(key, label_name)
                        try:
                            report[report_key] = result[key][l]
                        except IndexError:
                            report[report_key] = np.nan

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
