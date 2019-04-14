import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_semantic_segmentation
from chainercv.utils import apply_to_iterator


class SemanticSegmentationEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a semantic segmentation model.

    This extension iterates over an iterator and evaluates the prediction
    results of the model by common evaluation metrics for semantic
    segmentation.
    This extension reports values with keys below.
    Please note that :obj:`'iou/<label_names[l]>'` and
    :obj:`'class_accuracy/<label_names[l]>'` are reported only if
    :obj:`label_names` is specified.

    * :obj:`'miou'`: Mean of IoUs (mIoU).
    * :obj:`'iou/<label_names[l]>'`: IoU for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        For example, if :obj:`label_names` is \
        :obj:`~chainercv.datasets.camvid_label_names`, \
        this evaluator reports :obj:`'iou/Sky'`, \
        :obj:`'ap/Building'`, etc.
    * :obj:`'mean_class_accuracy'`: Mean of class accuracies.
    * :obj:`'class_accuracy/<label_names[l]>'`: Class accuracy for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class.
    * :obj:`'pixel_accuracy'`: Pixel accuracy.

    If there is no label assigned to class :obj:`label_names[l]`
    in the ground truth, values corresponding to keys
    :obj:`'iou/<label_names[l]>'` and :obj:`'class_accuracy/<label_names[l]>'`
    are :obj:`numpy.nan`.
    In that case, the means of them are calculated by excluding them from
    calculation.

    For details on the evaluation metrics, please see the documentation
    for :func:`chainercv.evaluations.eval_semantic_segmentation`.

    .. seealso::
        :func:`chainercv.evaluations.eval_semantic_segmentation`.

    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, label`.
            :obj:`img` is an image, :obj:`label` is pixel-wise label.
        target (chainer.Link): A semantic segmentation link. This link should
            have :meth:`predict` method that takes a list of images and
            returns :obj:`labels`.
        label_names (iterable of strings): An iterable of names of classes.
            If this value is specified, IoU and class accuracy for each class
            are also reported with the keys
            :obj:`'iou/<label_names[l]>'` and
            :obj:`'class_accuracy/<label_names[l]>'`.

    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, label_names=None):
        super(SemanticSegmentationEvaluator, self).__init__(
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

        pred_labels, = out_values
        gt_labels, = rest_values

        result = eval_semantic_segmentation(pred_labels, gt_labels)

        report = {'miou': result['miou'],
                  'pixel_accuracy': result['pixel_accuracy'],
                  'mean_class_accuracy': result['mean_class_accuracy']}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['iou/{:s}'.format(label_name)] = result['iou'][l]
                    report['class_accuracy/{:s}'.format(label_name)] =\
                        result['class_accuracy'][l]
                except IndexError:
                    report['iou/{:s}'.format(label_name)] = np.nan
                    report['class_accuracy/{:s}'.format(label_name)] = np.nan

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
