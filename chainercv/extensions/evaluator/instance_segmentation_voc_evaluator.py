import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_instance_segmentation_voc
from chainercv.utils import apply_to_iterator


class InstanceSegmentationVOCEvaluator(chainer.training.extensions.Evaluator):

    """An evaluation extension of instance-segmentation by PASCAL VOC metric.

    This extension iterates over an iterator and evaluates the prediction
    results by average precisions (APs) and mean of them
    (mean Average Precision, mAP).
    This extension reports the following values with keys.
    Please note that :obj:`'ap/<label_names[l]>'` is reported only if
    :obj:`label_names` is specified.

    * :obj:`'map'`: Mean of average precisions (mAP).
    * :obj:`'ap/<label_names[l]>'`: Average precision for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        For example, this evaluator reports :obj:`'ap/aeroplane'`, \
        :obj:`'ap/bicycle'`, etc. if :obj:`label_names` is \
        :obj:`~chainercv.datasets.sbd_instance_segmentation_label_names`. \
        If there is no bounding box assigned to class :obj:`label_names[l]` \
        in either ground truth or prediction, it reports :obj:`numpy.nan` as \
        its average precision. \
        In this case, mAP is computed without this class.

    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, bbox, label` or
            :obj:`img, bbox, label, difficult`.
            :obj:`img` is an image, :obj:`bbox` is coordinates of bounding
            boxes, :obj:`label` is labels of the bounding boxes and
            :obj:`difficult` is whether the bounding boxes are difficult or
            not. If :obj:`difficult` is returned, difficult ground truth
            will be ignored from evaluation.
        target (chainer.Link): An instance-segmentation link. This link must
            have :meth:`predict` method that takes a list of images and returns
            :obj:`bboxes`, :obj:`labels` and :obj:`scores`.
        iou_thresh (float): Intersection over Union (IoU) threshold for
            calulating average precision. The default value is 0.5.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
        label_names (iterable of strings): An iterable of names of classes.
            If this value is specified, average precision for each class is
            also reported with the key :obj:`'ap/<label_names[l]>'`.
    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target,
            iou_thresh=0.5, use_07_metric=False, label_names=None
    ):
        super(InstanceSegmentationVOCEvaluator, self).__init__(
            iterator, target)
        self.iou_thresh = iou_thresh
        self.use_07_metric = use_07_metric
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
        gt_masks, gt_labels = rest_values

        result = eval_instance_segmentation_voc(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels,
            iou_thresh=self.iou_thresh,
            use_07_metric=self.use_07_metric)

        report = {'map': result['map']}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
