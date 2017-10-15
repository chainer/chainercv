import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from eval_detection_coco import eval_detection_coco


class TestEvalDetectionCOCOAP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://github.com/yuyu2172/' \
            'share-weights/releases/download/0.0.5'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url,
            'coco_detection_dataset_val2014_fakebbox100_2017_10_15.npz'))[0])
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url,
            'coco_detection_result_val2014_fakebbox100_2017_10_15.npz'))[0])

    def test_eval_detection_voc(self):
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
            np.testing.assert_almost_equal(
                result[key], expected[key], decimal=5)


testing.run_module(__name__, __file__)
