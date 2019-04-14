from __future__ import division

import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from chainercv.evaluations import calc_instance_segmentation_voc_prec_rec
from chainercv.evaluations import eval_instance_segmentation_voc


@testing.parameterize(*(
    testing.product_dict(
        [{
            'pred_masks': [
                [[[False, False], [True, True]],
                 [[True, True], [False, True]],
                 [[True, False], [True, True]]]
            ],
            'pred_labels': [
                [0, 0, 0],
            ],
            'pred_scores': [
                [0.8, 0.9, 1],
            ],
            'gt_masks': [
                [[[True, True], [False, False]]],
            ],
            'gt_labels': [
                [0],
            ],
        }],
        [
            {
                'iou_thresh': 0.5,
                'prec': [
                    [0., 1 / 2, 1 / 3],
                ],
                'rec': [
                    [0., 1., 1.],
                ],
            },
            {
                'iou_thresh': 0.97,
                'prec': [
                    [0., 0., 0.],
                ],
                'rec': [
                    [0., 0., 0.],
                ],
            },
        ]
    ) +
    [
        {
            'pred_masks': [
                [[[False, False], [True, True]],
                 [[True, True], [False, False]]],
                [[[True, True], [True, True]],
                 [[True, True], [False, True]],
                 [[True, False], [True, True]]],
            ],
            'pred_labels': [
                [0, 0],
                [0, 2, 2],
            ],
            'pred_scores': [
                [1, 0.9],
                [0.7, 0.6, 0.8],
            ],
            'gt_masks': [
                [[[False, True], [True, True]],
                 [[True, True], [True, False]]],
                [[[True, False], [False, True]]],
            ],
            'gt_labels': [
                [0, 0],
                [2],
            ],
            'iou_thresh': 0.5,
            'prec': [
                [1., 1., 2 / 3],
                None,
                [1., 0.5],
            ],
            'rec': [
                [1 / 2, 1., 1.],
                None,
                [1., 1.],
            ],
        },
        {
            'pred_masks': [
                [[[False, True], [True, False]],
                 [[True, False], [False, True]],
                 [[True, False], [True, False]]]
            ],
            'pred_labels': [
                [0, 0, 0],
            ],
            'pred_scores': [
                [0.8, 0.9, 1],
            ],
            'gt_masks': [
                [[[False, True], [True, True]],
                 [[True, True], [False, True]]]
            ],
            'gt_labels': [
                [0, 0],
            ],
            'iou_thresh': 0.5,
            'prec': [
                [0, 1 / 2, 2 / 3],
            ],
            'rec': [
                [0, 1 / 2, 1.],
            ],
        },
    ]
))
class TestCalcInstanceSegmentationVOCPrecRec(unittest.TestCase):

    def setUp(self):
        self.pred_masks = (np.array(mask) for mask in self.pred_masks)
        self.pred_labels = (np.array(label) for label in self.pred_labels)
        self.pred_scores = (np.array(score) for score in self.pred_scores)
        self.gt_masks = (np.array(mask) for mask in self.gt_masks)
        self.gt_labels = (np.array(label) for label in self.gt_labels)

    def test_calc_instance_segmentation_voc_prec_rec(self):
        prec, rec = calc_instance_segmentation_voc_prec_rec(
            self.pred_masks, self.pred_labels, self.pred_scores,
            self.gt_masks, self.gt_labels,
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


class TestEvalInstanceSegmentationVOCAP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://chainercv-models.preferred.jp/tests'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url,
            'eval_instance_segmentation_voc_dataset_2018_04_04.npz'))[0],
            encoding='latin1')
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url,
            'eval_instance_segmentation_voc_result_2018_04_04.npz'))[0],
            encoding='latin1')

    def test_eval_instance_segmentation_voc(self):
        pred_masks = self.result['masks']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        gt_masks = self.dataset['masks']
        gt_labels = self.dataset['labels']

        result = eval_instance_segmentation_voc(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels,
            use_07_metric=True)

        # calculated from original python code
        expected = [
            0.159091,
            0.945455,
            0.679545,
            0.378293,
            0.430303,
            1.000000,
            0.581055,
            0.905195,
            0.415757,
            0.909091,
            1.000000,
            0.697256,
            0.856061,
            0.681818,
            0.793274,
            0.362141,
            0.948052,
            0.545455,
            0.840909,
            0.618182
        ]

        np.testing.assert_almost_equal(result['ap'], expected, decimal=5)
        np.testing.assert_almost_equal(
            result['map'], np.nanmean(expected), decimal=5)


testing.run_module(__name__, __file__)
