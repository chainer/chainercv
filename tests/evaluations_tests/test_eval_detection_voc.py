from __future__ import division

import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from chainercv.evaluations import calc_detection_voc_ap
from chainercv.evaluations import calc_detection_voc_prec_rec
from chainercv.evaluations import eval_detection_voc


@testing.parameterize(*(
    testing.product_dict(
        [{
            'pred_bboxes': [
                [[0, 0, 1, 1], [0, 0, 2, 2], [0.3, 0.3, 0.5, 0.5]],
            ],
            'pred_labels': [
                [0, 0, 0],
            ],
            'pred_scores': [
                [0.8, 0.9, 1],
            ],
            'gt_bboxes': [
                [[0, 0, 1, 0.9]],
            ],
            'gt_labels': [
                [0],
            ],
        }],
        [
            {
                'iou_thresh': 0.5,
                'prec': [
                    [0, 0, 1 / 3],
                ],
                'rec': [
                    [0, 0, 1],
                ],
            },
            {
                'iou_thresh': 0.97,
                'prec': [
                    [0, 0, 0],
                ],
                'rec': [
                    [0, 0, 0],
                ],
            },
        ]
    ) +
    [
        {
            'pred_bboxes': [
                [[0, 4, 1, 5], [0, 0, 1, 1]],
                [[0, 0, 2, 2], [2, 2, 3, 3], [5, 5, 7, 7]],
            ],
            'pred_labels': [
                [0, 0],
                [0, 2, 2],
            ],
            'pred_scores': [
                [1, 0.9],
                [0.7, 0.6, 0.8],
            ],
            'gt_bboxes': [
                [[0, 0, 1, 1], [1, 0, 4, 4]],
                [[2, 2, 3, 3]],
            ],
            'gt_labels': [
                [0, 0],
                [2],
            ],
            'iou_thresh': 0.4,
            'prec': [
                [0, 0.5, 1 / 3],
                None,
                [0, 0.5],
            ],
            'rec': [
                [0, 0.5, 0.5],
                None,
                [0, 1],
            ],
        },
        {
            'pred_bboxes': [
                [[0, 0, 1, 1], [0, 0, 2, 2], [0.3, 0.3, 0.5, 0.5]],
            ],
            'pred_labels': [
                [0, 0, 0],
            ],
            'pred_scores': [
                [0.8, 0.9, 1],
            ],
            'gt_bboxes': [
                [[0, 0, 1, 0.9], [1., 1., 2., 2.]],
            ],
            'gt_labels': [
                [0, 0],
            ],
            'gt_difficults': [
                [False, True],
            ],
            'iou_thresh': 0.5,
            'prec': [
                [0, 0, 1 / 3],
            ],
            'rec': [
                [0, 0, 1],
            ],
        },

    ]
))
class TestCalcDetectionVOCPrecRec(unittest.TestCase):

    def setUp(self):
        self.pred_bboxes = (np.array(bbox) for bbox in self.pred_bboxes)
        self.pred_labels = (np.array(label) for label in self.pred_labels)
        self.pred_scores = (np.array(score) for score in self.pred_scores)
        self.gt_bboxes = (np.array(bbox) for bbox in self.gt_bboxes)
        self.gt_labels = (np.array(label) for label in self.gt_labels)

        if hasattr(self, 'gt_difficults'):
            self.gt_difficults = (
                np.array(difficult) for difficult in self.gt_difficults)
        else:
            self.gt_difficults = None

    def test_calc_detection_voc_prec_rec(self):
        prec, rec = calc_detection_voc_prec_rec(
            self.pred_bboxes, self.pred_labels, self.pred_scores,
            self.gt_bboxes, self.gt_labels, self.gt_difficults,
            iou_thresh=self.iou_thresh)

        self.assertEqual(len(prec), len(self.prec))
        for prec_l, expected_prec_l in zip(prec, self.prec):
            if prec_l is None and expected_prec_l is None:
                continue
            np.testing.assert_equal(prec_l, expected_prec_l)

        self.assertEqual(len(rec), len(self.rec))
        for rec_l, expected_rec_l in zip(rec, self.rec):
            if rec_l is None and expected_rec_l is None:
                continue
            np.testing.assert_equal(rec_l, expected_rec_l)


@testing.parameterize(
    {'use_07_metric': False,
     'ap': [0.25, np.nan, 0.5]},
    {'use_07_metric': True,
     'ap': [0.5 / 11 * 6, np.nan, 0.5]},
)
class TestCalcDetectionVOCAP(unittest.TestCase):

    prec = [[0, 0.5, 1 / 3], None, [0, 0.5]]
    rec = [[0, 0.5, 0.5], None, [0, 1]]

    def setUp(self):
        self.prec = [
            np.array(prec_l) if prec_l is not None else None
            for prec_l in self.prec]
        self.rec = [
            np.array(rec_l) if rec_l is not None else None
            for rec_l in self.rec]

    def test_calc_detection_voc_ap(self):
        ap = calc_detection_voc_ap(
            self.prec, self.rec, use_07_metric=self.use_07_metric)

        np.testing.assert_almost_equal(ap, self.ap)


class TestEvalDetectionVOCAP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://github.com/yuyu2172/' \
            'share-weights/releases/download/0.0.3'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url,
            'voc_detection_dataset_2007_test_truncated_2017_06_06.npz'))[0])
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url,
            'voc_detection_result_2007_test_truncated_2017_06_06.npz'))[0])

    def test_eval_detection_voc(self):
        pred_bboxes = self.result['bboxes']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        gt_bboxes = self.dataset['bboxes']
        gt_labels = self.dataset['labels']
        gt_difficults = self.dataset['difficults']

        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)

        # these scores were calculated by MATLAB code
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

        np.testing.assert_almost_equal(result['ap'], expected, decimal=5)
        np.testing.assert_almost_equal(
            result['map'], np.nanmean(expected), decimal=5)
