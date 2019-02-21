import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from chainercv.datasets import coco_keypoint_names
from chainercv.evaluations import eval_keypoint_detection_coco

try:
    import pycocotools  # NOQA
    _available = True
except ImportError:
    _available = False


human_id = 0


def _generate_point(n_inst, size):
    H, W = size
    n_joint = len(coco_keypoint_names[human_id])
    ys = np.random.uniform(0, H, size=(n_inst, n_joint))
    xs = np.random.uniform(0, W, size=(n_inst, n_joint))
    point = np.stack((ys, xs), axis=2).astype(np.float32)

    valid = np.random.randint(0, 2, size=(n_inst, n_joint)).astype(np.bool)
    return point, valid


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestEvalPointCOCOSingleClass(unittest.TestCase):

    n_inst = 3

    def setUp(self):
        self.pred_points = []
        self.pred_labels = []
        self.pred_scores = []
        self.gt_points = []
        self.gt_visibles = []
        self.gt_bboxes = []
        self.gt_labels = []
        for i in range(2):
            point, valid = _generate_point(self.n_inst, (32, 48))
            self.pred_points.append(point)
            self.pred_labels.append(np.zeros((self.n_inst,), dtype=np.int32))
            self.pred_scores.append(np.random.uniform(
                0.5, 1, size=(self.n_inst,)).astype(np.float32))
            self.gt_points.append(point)
            self.gt_visibles.append(valid)
            bbox = np.zeros((self.n_inst, 4), dtype=np.float32)
            for i, pnt in enumerate(point):
                y_min = np.min(pnt[:, 0])
                x_min = np.min(pnt[:, 1])
                y_max = np.max(pnt[:, 0])
                x_max = np.max(pnt[:, 1])
                bbox[i] = [y_min, x_min, y_max, x_max]
            self.gt_bboxes.append(bbox)
            self.gt_labels.append(np.zeros((self.n_inst,), dtype=np.int32))

    def _check(self, result):
        self.assertEqual(result['map/iou=0.50:0.95/area=all/max_dets=20'], 1)
        self.assertEqual(result['map/iou=0.50/area=all/max_dets=20'], 1)
        self.assertEqual(result['map/iou=0.75/area=all/max_dets=20'], 1)
        self.assertEqual(result['mar/iou=0.50:0.95/area=all/max_dets=20'], 1)
        self.assertEqual(result['mar/iou=0.50/area=all/max_dets=20'], 1)
        self.assertEqual(result['mar/iou=0.75/area=all/max_dets=20'], 1)

    def test_gt_bboxes_not_supplied(self):
        result = eval_keypoint_detection_coco(
            self.pred_points, self.pred_labels, self.pred_scores,
            self.gt_points, self.gt_visibles, None, self.gt_labels)
        self._check(result)

    def test_area_not_supplied(self):
        result = eval_keypoint_detection_coco(
            self.pred_points, self.pred_labels, self.pred_scores,
            self.gt_points, self.gt_visibles, self.gt_bboxes, self.gt_labels)
        self._check(result)

        self.assertFalse(
            'map/iou=0.50:0.95/area=medium/max_dets=20' in result)
        self.assertFalse(
            'map/iou=0.50:0.95/area=large/max_dets=20' in result)
        self.assertFalse(
            'mar/iou=0.50:0.95/area=medium/max_dets=20' in result)
        self.assertFalse(
            'mar/iou=0.50:0.95/area=large/max_dets=20' in result)

    def test_area_supplied(self):
        gt_areas = [[100] * self.n_inst for _ in range(2)]
        result = eval_keypoint_detection_coco(
            self.pred_points, self.pred_labels, self.pred_scores,
            self.gt_points, self.gt_visibles, self.gt_bboxes, self.gt_labels,
            gt_areas=gt_areas,
        )
        self._check(result)
        self.assertTrue(
            'map/iou=0.50:0.95/area=medium/max_dets=20' in result)
        self.assertTrue(
            'map/iou=0.50:0.95/area=large/max_dets=20' in result)
        self.assertTrue(
            'mar/iou=0.50:0.95/area=medium/max_dets=20' in result)
        self.assertTrue(
            'mar/iou=0.50:0.95/area=large/max_dets=20' in result)

    def test_crowded_supplied(self):
        gt_crowdeds = [[True] * self.n_inst for _ in range(2)]
        result = eval_keypoint_detection_coco(
            self.pred_points, self.pred_labels, self.pred_scores,
            self.gt_points, self.gt_visibles, self.gt_bboxes, self.gt_labels,
            gt_crowdeds=gt_crowdeds,
        )
        # When the only ground truth is crowded, nothing is evaluated.
        # In that case, all the results are nan.
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=all/max_dets=20']))


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestEvalKeypointDetectionCOCO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://chainercv-models.preferred.jp/tests'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url,
            'eval_keypoint_detection_coco_dataset_2019_02_21.npz'))[0])
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url,
            'eval_keypoint_detection_coco_result_2019_02_20.npz'))[0])

    def test_eval_keypoint_detection_coco(self):
        pred_points = self.result['points']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        gt_points = self.dataset['points']
        gt_visibles = self.dataset['visibles']
        gt_bboxes = self.dataset['bboxes']
        gt_labels = self.dataset['labels']
        gt_areas = self.dataset['areas']
        gt_crowdeds = self.dataset['crowdeds']

        result = eval_keypoint_detection_coco(
            pred_points, pred_labels, pred_scores,
            gt_points, gt_visibles, gt_bboxes,
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
