import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from chainercv.evaluations import eval_detection_coco

try:
    import pycocotools  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


data = {
    'pred_bboxes': [
        [[0, 0, 10, 10], [0, 0, 20, 20]]],
    'pred_labels': [
        [0, 0]],
    'pred_scores': [
        [0.8, 0.9]],
    'gt_bboxes': [
        [[0, 0, 10, 9]]],
    'gt_labels': [
        [0, 0]]}


class TestEvalDetectionCOCOSimple(unittest.TestCase):

    def setUp(self):
        self.pred_bboxes = (np.array(bbox) for bbox in data['pred_bboxes'])
        self.pred_labels = (np.array(label) for label in data['pred_labels'])
        self.pred_scores = (np.array(score) for score in data['pred_scores'])
        self.gt_bboxes = (np.array(bbox) for bbox in data['gt_bboxes'])
        self.gt_labels = (np.array(label) for label in data['gt_labels'])

    def test_crowded(self):
        if not optional_modules:
            return
        result = eval_detection_coco(self.pred_bboxes, self.pred_labels,
                                     self.pred_scores,
                                     self.gt_bboxes, self.gt_labels,
                                     gt_crowdeds=[[True]])
        # When the only ground truth is crowded, nothing is evaluated.
        # In that case, all the results are nan.
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=small/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=medium/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=large/maxDets=100']))

    def test_area_default(self):
        if not optional_modules:
            return
        result = eval_detection_coco(self.pred_bboxes, self.pred_labels,
                                     self.pred_scores,
                                     self.gt_bboxes, self.gt_labels)
        # Test that the original bbox area is used, which is 90.
        # In that case, the ground truth bounding box is assigned to segment
        # "small".
        # Therefore, the score for segments "medium" and "large" will be nan.
        self.assertFalse(
            np.isnan(result['map/iou=0.50:0.95/area=small/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=medium/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=large/maxDets=100']))

    def test_area_specified(self):
        if not optional_modules:
            return
        result = eval_detection_coco(self.pred_bboxes, self.pred_labels,
                                     self.pred_scores,
                                     self.gt_bboxes, self.gt_labels,
                                     gt_areas=[[2048]]
                                     )
        self.assertFalse(
            np.isnan(result['map/iou=0.50:0.95/area=medium/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=small/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=large/maxDets=100']))


class TestEvalDetectionCOCO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://github.com/yuyu2172/' \
            'share-weights/releases/download/0.0.5'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url,
            'coco_detection_dataset_val2014_fakebbox100_2017_10_16.npz'))[0])
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url,
            'coco_detection_result_val2014_fakebbox100_2017_10_16.npz'))[0])

    def test_eval_detection_voc(self):
        if not optional_modules:
            return
        pred_bboxes = self.result['bboxes']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        gt_bboxes = self.dataset['bboxes']
        gt_labels = self.dataset['labels']
        gt_crowdeds = self.dataset['crowdeds']
        gt_areas = self.dataset['areas']

        result = eval_detection_coco(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_crowdeds, gt_areas)

        expected = {
            'map/iou=0.50:0.95/area=all/maxDets=100': 0.5069852,
            'map/iou=0.50/area=all/maxDets=100': 0.69937725,
            'map/iou=0.75/area=all/maxDets=100': 0.57538619,
            'map/iou=0.50:0.95/area=small/maxDets=100': 0.58562572,
            'map/iou=0.50:0.95/area=medium/maxDets=100': 0.51939969,
            'map/iou=0.50:0.95/area=large/maxDets=100': 0.5013979,
            'mar/iou=0.50:0.95/area=all/maxDets=1': 0.38919373,
            'mar/iou=0.50:0.95/area=all/maxDets=10': 0.59606053,
            'mar/iou=0.50:0.95/area=all/maxDets=100': 0.59773394,
            'mar/iou=0.50:0.95/area=small/maxDets=100': 0.63981096,
            'mar/iou=0.50:0.95/area=medium/maxDets=100': 0.5664206,
            'mar/iou=0.50:0.95/area=large/maxDets=100': 0.5642906
        }

        for key, item in expected.items():
            non_mean_key = key[1:]
            self.assertIsInstance(result[non_mean_key], np.ndarray)
            self.assertEqual(result[non_mean_key].shape, (76,))
            np.testing.assert_almost_equal(
                result[key], expected[key], decimal=5)


testing.run_module(__name__, __file__)
