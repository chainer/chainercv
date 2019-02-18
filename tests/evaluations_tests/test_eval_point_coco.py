import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from chainercv.evaluations import eval_point_coco

try:
    import pycocotools  # NOQA
    _available = True
except ImportError:
    _available = False


# @unittest.skipUnless(_available, 'pycocotools is not installed')
# class TestEvalPointCOCOSingleClass(unittest.TestCase):
# 
#     def setUp(self):
#         self.pred_bboxes = np.array([[[0, 0, 10, 10], [0, 0, 20, 20]]])
#         self.pred_labels = np.array([[0, 0]])
#         self.pred_scores = np.array([[0.8, 0.9]])
#         self.gt_bboxes = np.array([[[0, 0, 10, 9]]])
#         self.gt_labels = np.array([[0, 0]])
# 
#     def test_crowded(self):
#         result = eval_detection_coco(self.pred_bboxes, self.pred_labels,
#                                      self.pred_scores,
#                                      self.gt_bboxes, self.gt_labels,
#                                      gt_crowdeds=[[True]])
#         # When the only ground truth is crowded, nothing is evaluated.
#         # In that case, all the results are nan.
#         self.assertTrue(
#             np.isnan(result['map/iou=0.50:0.95/area=all/max_dets=100']))
#         self.assertTrue(
#             np.isnan(result['map/iou=0.50/area=all/max_dets=100']))
#         self.assertTrue(
#             np.isnan(result['map/iou=0.75/area=all/max_dets=100']))
# 
#     def test_area_not_supplied(self):
#         result = eval_detection_coco(self.pred_bboxes, self.pred_labels,
#                                      self.pred_scores,
#                                      self.gt_bboxes, self.gt_labels)
#         self.assertFalse(
#             'map/iou=0.50:0.95/area=small/max_dets=100' in result)
#         self.assertFalse(
#             'map/iou=0.50:0.95/area=medium/max_dets=100' in result)
#         self.assertFalse(
#             'map/iou=0.50:0.95/area=large/max_dets=100' in result)
# 
#     def test_area_specified(self):
#         result = eval_detection_coco(self.pred_bboxes, self.pred_labels,
#                                      self.pred_scores,
#                                      self.gt_bboxes, self.gt_labels,
#                                      gt_areas=[[2048]])
#         self.assertFalse(
#             np.isnan(result['map/iou=0.50:0.95/area=medium/max_dets=100']))
#         self.assertTrue(
#             np.isnan(result['map/iou=0.50:0.95/area=small/max_dets=100']))
#         self.assertTrue(
#             np.isnan(result['map/iou=0.50:0.95/area=large/max_dets=100']))


# @unittest.skipUnless(_available, 'pycocotools is not installed')
# class TestEvalPointCOCOSomeClassNonExistent(unittest.TestCase):
# 
#     def setUp(self):
#         self.pred_bboxes = np.array([[[0, 0, 10, 10], [0, 0, 20, 20]]])
#         self.pred_labels = np.array([[1, 2]])
#         self.pred_scores = np.array([[0.8, 0.9]])
#         self.gt_bboxes = np.array([[[0, 0, 10, 9]]])
#         self.gt_labels = np.array([[1, 2]])
# 
#     def test(self):
#         result = eval_detection_coco(self.pred_bboxes, self.pred_labels,
#                                      self.pred_scores,
#                                      self.gt_bboxes, self.gt_labels)
#         self.assertEqual(
#             result['ap/iou=0.50:0.95/area=all/max_dets=100'].shape, (3,))
#         self.assertTrue(
#             np.isnan(result['ap/iou=0.50:0.95/area=all/max_dets=100'][0]))
#         self.assertEqual(
#             np.nanmean(result['ap/iou=0.50:0.95/area=all/max_dets=100'][1:]),
#             result['map/iou=0.50:0.95/area=all/max_dets=100'])
# 

@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestEvalPointCOCO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://chainercv-models.preferred.jp/tests'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url, 'eval_point_coco_dataset_2019_02_18.npz'))[0])
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url, 'eval_point_coco_result_2019_02_18.npz'))[0])

    def test_eval_detection_coco(self):
        pred_points = self.result['points']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        gt_points = self.dataset['points']
        gt_is_valids = self.dataset['is_valids']
        gt_bboxes = self.dataset['bboxes']
        gt_labels = self.dataset['labels']
        gt_areas = self.dataset['areas']
        gt_crowdeds = self.dataset['crowdeds']

        result = eval_point_coco(
            pred_points, pred_labels, pred_scores,
            gt_points, gt_is_valids, gt_bboxes,
            gt_labels, gt_areas, gt_crowdeds)


        expected = {
            'map/iou=0.50:0.95/area=all/max_dets=20': 0.37733572721481323,
            'map/iou=0.50/area=all/max_dets=20': 0.6448841691017151,
            'map/iou=0.75/area=all/max_dets=20': 0.35469090938568115,
            'map/iou=0.50:0.95/area=medium/max_dets=20': 0.3894105851650238,
            'map/iou=0.50:0.95/area=large/max_dets=20': 0.39169296622276306,
            'mar/iou=0.50:0.95/area=all/max_dets=20': 0.5218977928161621,
            'mar/iou=0.50/area=all/max_dets=20': 0.7445255517959595,
            'mar/iou=0.75/area=all/max_dets=20': 0.510948896408081,
            'mar/iou=0.50:0.95/area=medium/max_dets=20': 0.5150684714317322,
            'mar/iou=0.50:0.95/area=large/max_dets=20': 0.5296875238418579,
        }

        for key, item in expected.items():
            np.testing.assert_almost_equal(
                result[key], expected[key], decimal=5)


testing.run_module(__name__, __file__)
