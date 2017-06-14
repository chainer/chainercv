import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_semantic_segmentation_iou
from chainercv.utils import apply_prediction_to_iterator


class SemanticSegmentationEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a semantic segmentation model.

    This extension iterates over an iterator and evaluates the prediction
    results of the model by Intersection over Union (IoU) for each class and
    the mean of the IoUs (mIoU).
    This extension reports the following values with keys.
    Please note that :obj:`'iou/<label_names[l]>'` is reported only if
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
        In this case, IoU is computed without this class.

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

        iou = eval_semantic_segmentation_iou(pred_labels, gt_labels)

        report = {'miou': np.nanmean(iou)}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['iou/{:s}'.format(label_name)] = iou[l]
                except IndexError:
                    report['iou/{:s}'.format(label_name)] = np.nan

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
