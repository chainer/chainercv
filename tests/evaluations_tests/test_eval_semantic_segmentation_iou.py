from __future__ import division

import unittest

import numpy as np

from chainer import testing

from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.evaluations import calc_semantic_segmentation_iou
from chainercv.evaluations import eval_semantic_segmentation_iou


def _pred_iterator():
    pred_labels = np.repeat([[[1, 1, 0], [0, 0, 1]]], 2, axis=0)
    for pred_label in pred_labels:
        yield pred_label


def _gt_iterator():
    gt_labels = np.repeat([[[1, 0, 0], [0, -1, 1]]], 2, axis=0)
    for gt_label in gt_labels:
        yield gt_label


@testing.parameterize(
    {'pred_labels': _pred_iterator(),
     'gt_labels': _gt_iterator(),
     'iou': np.array([4. / 6., 4. / 6.])
     },
    {'pred_labels': np.array([[[0, 0, 0], [0, 0, 0]]]),
     'gt_labels': np.array([[[1, 1, 1], [1, 1, 1]]]),
     'iou': np.array([0, 0]),
     }
)
class TestEvalSemanticSegmentationIou(unittest.TestCase):

    n_class = 2

    def test_eval_semantic_segmentation_iou(self):
        iou = eval_semantic_segmentation_iou(
            self.pred_labels, self.gt_labels, self.n_class)
        np.testing.assert_equal(iou, self.iou)


class TestCalcSemanticSegmentationConfusion(unittest.TestCase):

    n_class = 2

    def test_calc_semantic_segmentation_confusion(self):
        pred_labels = np.random.randint(0, self.n_class, size=(10, 16, 16))
        gt_labels = np.random.randint(-1, self.n_class, size=(10, 16, 16))
        expected = np.zeros((self.n_class, self.n_class), dtype=np.int64)
        expected[0, 0] = np.sum(
            np.logical_and(gt_labels == 0, pred_labels == 0))
        expected[0, 1] = np.sum(
            np.logical_and(gt_labels == 0, pred_labels == 1))
        expected[1, 0] = np.sum(
            np.logical_and(gt_labels == 1, pred_labels == 0))
        expected[1, 1] = np.sum(
            np.logical_and(gt_labels == 1, pred_labels == 1))

        confusion = calc_semantic_segmentation_confusion(
            pred_labels, gt_labels, self.n_class)
        np.testing.assert_equal(confusion, expected)


class TestCalcSemanticSegmentationIou(unittest.TestCase):

    n_class = 2

    def test_calc_semantic_segmentation_iou(self):
        c = np.random.randint(0, 100, size=(self.n_class, self.n_class))
        expected = np.array(
            [c[0, 0] / (c[0, 0] + c[0, 1] + c[1, 0]),
             c[1, 1] / (c[1, 1] + c[1, 0] + c[0, 1])])

        iou = calc_semantic_segmentation_iou(c)
        np.testing.assert_equal(iou, expected)


testing.run_module(__name__, __file__)
