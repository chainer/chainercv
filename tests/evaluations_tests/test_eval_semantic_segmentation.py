from __future__ import division

import unittest

import numpy as np

from chainer import testing

from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.evaluations import calc_semantic_segmentation_iou
from chainercv.evaluations import eval_semantic_segmentation


@testing.parameterize(
    {'pred_labels': iter(np.repeat([[[1, 1, 0], [0, 0, 1]]], 2, axis=0)),
     'gt_labels': iter(np.repeat([[[1, 0, 0], [0, -1, 1]]], 2, axis=0)),
     'iou': np.array([4 / 6, 4 / 6]),
     'pixel_accuracy': 4 / 5,
     'class_accuracy': np.array([2 / 3, 2 / 2]),
     },
    {'pred_labels': np.array([[[0, 0, 0], [0, 0, 0]]]),
     'gt_labels': np.array([[[1, 1, 1], [1, 1, 1]]]),
     'iou': np.array([0, 0]),
     'pixel_accuracy': 0 / 6,
     'class_accuracy': np.array([np.nan, 0])
     }
)
class TestEvalSemanticSegmentation(unittest.TestCase):

    def test_eval_semantic_segmentation(self):
        result = eval_semantic_segmentation(
            self.pred_labels, self.gt_labels)
        np.testing.assert_equal(result['iou'], self.iou)
        np.testing.assert_equal(result['pixel_accuracy'], self.pixel_accuracy)
        np.testing.assert_equal(result['class_accuracy'], self.class_accuracy)

        np.testing.assert_equal(result['miou'], np.nanmean(self.iou))
        np.testing.assert_equal(
            result['mean_class_accuracy'], np.nanmean(self.class_accuracy))


class TestCalcSemanticSegmentationConfusion(unittest.TestCase):

    def test_calc_semantic_segmentation_confusion(self):
        n_class = 2
        pred_labels = np.random.randint(0, n_class, size=(10, 16, 16))
        gt_labels = np.random.randint(-1, n_class, size=(10, 16, 16))
        expected = np.zeros((n_class, n_class), dtype=np.int64)
        expected[0, 0] = np.sum(
            np.logical_and(gt_labels == 0, pred_labels == 0))
        expected[0, 1] = np.sum(
            np.logical_and(gt_labels == 0, pred_labels == 1))
        expected[1, 0] = np.sum(
            np.logical_and(gt_labels == 1, pred_labels == 0))
        expected[1, 1] = np.sum(
            np.logical_and(gt_labels == 1, pred_labels == 1))

        confusion = calc_semantic_segmentation_confusion(
            pred_labels, gt_labels)
        np.testing.assert_equal(confusion, expected)

    def test_calc_semantic_segmentation_confusion_shape(self):
        n_class = 30
        pred_labels = np.random.randint(0, n_class, size=(2, 3, 3))
        gt_labels = np.random.randint(-1, n_class, size=(2, 3, 3))
        confusion = calc_semantic_segmentation_confusion(
            pred_labels, gt_labels)

        size = (np.max((pred_labels + 1, gt_labels + 1)))
        self.assertEqual(confusion.shape, (size, size))


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
