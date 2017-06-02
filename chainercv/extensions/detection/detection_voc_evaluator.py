import copy

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_detection_voc_ap
from chainercv.utils import apply_prediction_to_iterator


class DetectionVOCEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a detection model by PASCAL VOC metric.

    This extension iterates over an iterator and evaluates the prediction
    results of the model by PASCAL VOC's mAP metrics.

    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, bbox, label` or
            :obj:`img, bbox, label, difficult`.
            :obj:`img` is an image, :obj:`bbox` is coordinates of bounding
            boxes, :obj:`label` is labels of the bounding boxes and
            :obj:`difficult` is whether the bounding boxes are difficult or
            not. If :obj:`difficult` is returned, difficult ground truth
            will be ignored from evaluation.
        target (chainer.Link): An detection link. This link must have
            :meth:`predict` method which takes a list of images and returns
            :obj:`bboxes`, :obj:`labels` and :obj:`scores`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, use_07_metric=False):
        super(DetectionVOCEvaluator, self).__init__(
            iterator, target)
        self.use_07_metric = use_07_metric

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterator explicitly
        del imgs

        pred_bboxes, pred_labels, pred_scores = pred_values

        if len(gt_values) == 3:
            gt_bboxes, gt_labels, gt_difficults = gt_values
        elif len(gt_values) == 2:
            gt_bboxes, gt_labels = gt_values
            gt_difficults = None

        ap = eval_detection_voc_ap(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=self.use_07_metric)
        map_ = sum(ap_l for ap_l in ap if ap_l is not None)

        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'map': map_}, target)
        return observation
