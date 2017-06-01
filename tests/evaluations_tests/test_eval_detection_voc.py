import numpy as np
import os
import unittest
from six.moves.urllib import request

from chainer import testing

from chainercv.evaluations import eval_detection_voc


@testing.parameterize(
    {'iou_thresh': 0.5,
     'rec': np.array([0., 0., 1.]),
     'prec': np.array([0., 0., 1. / 3.])},
    {'iou_thresh': 0.97,
     'rec': np.array([0., 0., 0.]),
     'prec': np.array([0., 0., 0.])}
)
class TestEvalDetectionVOCOneBbox(unittest.TestCase):

    def test_eval_detection_voc_one_bbox(self):
        pred_bboxes = iter([np.array([
            [0., 0., 1., 1.], [0., 0., 2., 2.], [0.3, 0.3, 0.5, 0.5]])])
        pred_labels = iter([np.array([0, 0, 0])])
        pred_scores = iter([np.array([0.8, 0.9, 1.])])
        gt_bboxes = iter([np.array([[0., 0., 1., 0.9]])])
        gt_labels = iter([np.array([0])])
        # iou is [0.95, 0.422, 0.3789]

        results = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
            iou_thresh=self.iou_thresh)
        np.testing.assert_equal(results[0]['recall'], self.rec)
        np.testing.assert_equal(results[0]['precision'], self.prec)


@testing.parameterize(
    {'use_07_metric': False,
     'ap0': 0.25,
     'ap1': 0.5},
    {'use_07_metric': True,
     'ap0': 0.5 / 11. * 6,
     'ap1': 0.5},
)
class TestEvalDetectionVOCMultipleBboxes(unittest.TestCase):

    iou_thresh = 0.4
    rec0 = np.array([0.0, 0.5, 0.5])
    prec0 = np.array([0., 0.5, 1. / 3.])
    rec1 = np.array([0., 1.])
    prec1 = np.array([0., 0.5])

    def test_eval_detection_voc(self):
        pred_bboxes = iter([
            np.array([[0., 4., 1., 5.], [0., 0., 1., 1.]]),
            np.array([[0., 0., 2., 2.], [2., 2., 3., 3.], [5., 5., 7., 7.]])
        ])
        pred_labels = iter([np.array([0, 0]), np.array([0, 1, 1])])
        pred_scores = iter([np.array([1., 0.9]), np.array([0.7, 0.6, 0.8])])
        gt_bboxes = iter([
            np.array([[0., 0., 1., 1.], [1., 0., 4., 4.]]),
            np.array([[2., 2., 3., 3.]])
        ])
        gt_labels = iter([np.array([0, 0]), np.array([1])])

        results = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
            iou_thresh=self.iou_thresh,
            use_07_metric=self.use_07_metric)
        np.testing.assert_equal(results[0]['recall'], self.rec0)
        np.testing.assert_equal(results[0]['precision'], self.prec0)
        np.testing.assert_almost_equal(results[0]['ap'], self.ap0)
        np.testing.assert_equal(results[1]['recall'], self.rec1)
        np.testing.assert_equal(results[1]['precision'], self.prec1)
        np.testing.assert_almost_equal(results[1]['ap'], self.ap1)
        np.testing.assert_almost_equal(
            results['map'], (self.ap0 + self.ap1) / 2)


class TestEvalDetectionVOCDifficults(unittest.TestCase):

    iou_thresh = 0.5
    rec = np.array([0., 0., 1.])
    prec = np.array([0., 0., 1. / 3.])

    def test_eval_detection_voc_difficult(self):
        pred_bboxes = iter([np.array([
            [0., 0., 1., 1.], [0., 0., 2., 2.], [0.3, 0.3, 0.5, 0.5]])])
        pred_labels = iter([np.array([0, 0, 0])])
        pred_scores = iter([np.array([0.8, 0.9, 1.])])
        gt_bboxes = iter([np.array([[0., 0., 1., 0.9], [1., 1., 2., 2.]])])
        gt_labels = iter([np.array([0, 0])])
        gt_difficults = iter([np.array([False, True])])
        # iou is [0.95, 0.422, 0.3789] and [0.142, 0.444, 0.048]

        results = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
            gt_difficults=gt_difficults, iou_thresh=self.iou_thresh)
        np.testing.assert_equal(results[0]['recall'], self.rec)
        np.testing.assert_equal(results[0]['precision'], self.prec)


class TestEvalDetectionVOCConsistencyWithMATLAB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://github.com/yuyu2172/' \
                   'share-weights/releases/download/0.0.2'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url, 'example_detection_dataset_2017_06_01.npz'))[0])
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url, 'example_detection_result_2017_06_01.npz'))[0])

    def test_eval_detection_voc_consistency_with_matlab(self):
        pred_bboxes = self.result['bboxes']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        gt_bboxes = self.dataset['bboxes']
        gt_labels = self.dataset['labels']
        gt_difficults = self.dataset['difficults']

        eval_ = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)

        expected = [
            0.772727,
            0.738780,
            0.957576,
            0.640153,
            0.579473,
            1.000000,
            0.970030,
            1.000000,
            0.705931,
            0.678719,
            0.863636,
            1.000000,
            1.000000,
            0.561364,
            0.798813,
            0.712121,
            0.939394,
            0.563636,
            0.927273,
            0.654545,
        ]

        np.testing.assert_almost_equal(
            [eval_[l]['ap'] for l in range(20)], expected, decimal=5)
