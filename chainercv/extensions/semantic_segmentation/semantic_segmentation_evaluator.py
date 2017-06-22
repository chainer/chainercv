import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_semantic_segmentation
from chainercv.utils import apply_prediction_to_iterator


class SemanticSegmentationEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a semantic segmentation model.

    This extension iterates over an iterator and evaluates the prediction
    results of the model by common evaluation metrics for semantic
    segmentation.
    This extension reports the following values with keys.
    Please note that :obj:`'iou/<label_names[l]>'` and
    :obj:`'class_accuracy/<label_names[l]>'` are reported only if
    :obj:`label_names` is specified.

    * :obj:`'miou'`: Mean of IoUs (mIoU).
    * :obj:`'iou/<label_names[l]>'`: IoU for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        For example, this evaluator reports :obj:`'iou/Sky'`, \
        :obj:`'ap/Building'`, etc. if :obj:`label_names` is \
        :obj:`~chainercv.datasets.camvid_label_names`. \
        If there is no label assigned to class :obj:`label_names[l]` \
        in ground truth, it reports :obj:`numpy.nan` as \
        its IoU. \
        In this case, mean IoU is computed without this class.
    * :obj:`'mean_class_accuracy'`: Mean of class accuracies.
    * :obj:`class_accuracy/<label_names[l]>'`: Class accuracy for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        If there is no label assigned to class :obj:`label_names[l]` \
        in ground truth, it reports :obj:`numpy.nam` as \
        its class accuracy. \
        In this case, mean class accuracy is computed without this class.
    * :obj:`pixel_accuracy`: Pixel accuracy.

    For details on the evaluation metrics, please see the documentation
    for :func:`chainercv.evaluations.eval_semantic_segmentation`.

    .. seealso::
        :func:`chainercv.evaluations.eval_semantic_segmentation`.

    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, label`.
            :obj:`img` is an image, :obj:`label` is pixel-wise label.
        target (chainer.Link): A semantic segmentation link. This link should
            have :meth:`predict` method which takes a list of images and
            returns :obj:`labels`.
        label_names (iterable of strings): An iterable of names of classes.
            If this value is specified, IoU for each class is
            also reported with the key :obj:`'iou/<label_names[l]>'`.

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

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterator explicitly
        del imgs

        pred_labels, = pred_values
        gt_labels, = gt_values

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
