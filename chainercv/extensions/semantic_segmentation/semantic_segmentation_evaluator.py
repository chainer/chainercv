import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_semantic_segmentation_iou
from chainercv.utils import apply_prediction_to_iterator


class SemanticSegmentationEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a semantic segmentation model.

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

    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, n_class, label_names=None):
        super(SemanticSegmentationEvaluator, self).__init__(
            iterator, target)
        self.n_class = n_class

        if label_names is not None and len(label_names) != n_class:
            raise ValueError('The number of classes and the length of'
                             'label_names should be same.')
        if label_names is None:
            label_names = tuple(range(n_class))
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

        ious = eval_semantic_segmentation_iou(
            pred_labels, gt_labels, self.n_class)

        observation = {}
        with reporter.report_scope(observation):
            for label_name, iou in zip(self.label_names, ious):
                reporter.report({'{}/iou'.format(label_name): iou}, target)
            reporter.report({'miou': np.nanmean(ious)}, target)
        return observation
