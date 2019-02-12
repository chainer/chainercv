import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from chainercv.evaluations import eval_instance_segmentation_coco

try:
    import pycocotools  # NOQA
    _available = True
except ImportError:
    _available = False


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestEvalInstanceSegmentationCOCOSimple(unittest.TestCase):

    def setUp(self):
        self.pred_masks = np.array(
            [[[[True, True], [True, True]],
              [[True, False], [False, True]]]])
        self.pred_labels = np.array([[0, 0]])
        self.pred_scores = np.array([[0.8, 0.9]])
        self.gt_masks = np.array([[[[True, True], [True, True]]]])
        self.gt_labels = np.array([[0, 0]])

    def test_crowded(self):
        result = eval_instance_segmentation_coco(
            self.pred_masks, self.pred_labels, self.pred_scores,
            self.gt_masks, self.gt_labels, gt_crowdeds=[[True]])
        # When the only ground truth is crowded, nothing is evaluated.
        # In that case, all the results are nan.
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=all/max_dets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50/area=all/max_dets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.75/area=all/max_dets=100']))

    def test_area_not_supplied(self):
        result = eval_instance_segmentation_coco(
            self.pred_masks, self.pred_labels, self.pred_scores,
            self.gt_masks, self.gt_labels)
        self.assertFalse(
            'map/iou=0.50:0.95/area=small/max_dets=100' in result)
        self.assertFalse(
            'map/iou=0.50:0.95/area=medium/max_dets=100' in result)
        self.assertFalse(
            'map/iou=0.50:0.95/area=large/max_dets=100' in result)

    def test_area_specified(self):
        result = eval_instance_segmentation_coco(
            self.pred_masks, self.pred_labels, self.pred_scores,
            self.gt_masks, self.gt_labels, gt_areas=[[2048]])
        self.assertFalse(
            np.isnan(result['map/iou=0.50:0.95/area=medium/max_dets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=small/max_dets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=large/max_dets=100']))


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestEvalInstanceSegmentationCOCOSomeClassNonExistent(unittest.TestCase):

    def setUp(self):
        self.pred_masks = np.array(
            [[[[True, True], [True, True]],
              [[True, False], [False, True]]]])
        self.pred_labels = np.array([[1, 2]])
        self.pred_scores = np.array([[0.8, 0.9]])
        self.gt_masks = np.array([[[[True, True], [True, True]]]])
        self.gt_labels = np.array([[1, 2]])

    def test(self):
        result = eval_instance_segmentation_coco(
            self.pred_masks, self.pred_labels, self.pred_scores,
            self.gt_masks, self.gt_labels)
        self.assertEqual(
            result['ap/iou=0.50:0.95/area=all/max_dets=100'].shape, (3,))
        self.assertTrue(
            np.isnan(result['ap/iou=0.50:0.95/area=all/max_dets=100'][0]))
        self.assertEqual(
            np.nanmean(result['ap/iou=0.50:0.95/area=all/max_dets=100'][1:]),
            result['map/iou=0.50:0.95/area=all/max_dets=100'])


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestEvalInstanceSegmentationCOCOEmptyPred(unittest.TestCase):

    def setUp(self):
        self.pred_masks = np.zeros((1, 0, 2, 2), dtype=np.bool)
        self.pred_labels = np.zeros((1, 0), dtype=np.int32)
        self.pred_scores = np.zeros((1, 0), dtype=np.float32)
        self.gt_masks = np.array([[[[True, True], [True, True]]]])
        self.gt_labels = np.array([[1, 2]])

    def test(self):
        result = eval_instance_segmentation_coco(
            self.pred_masks, self.pred_labels, self.pred_scores,
            self.gt_masks, self.gt_labels)
        self.assertEqual(
            result['ap/iou=0.50:0.95/area=all/max_dets=100'].shape, (2,))
        self.assertTrue(
            np.isnan(result['ap/iou=0.50:0.95/area=all/max_dets=100'][0]))
        self.assertEqual(
            np.nanmean(result['ap/iou=0.50:0.95/area=all/max_dets=100'][1:]),
            result['map/iou=0.50:0.95/area=all/max_dets=100'])


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestEvalInstanceSegmentationCOCO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://chainercv-models.preferred.jp/tests'

        cls.dataset = np.load(request.urlretrieve(os.path.join(
            base_url,
            'eval_instance_segmentation_coco_dataset_2018_07_06.npz'))[0],
            encoding='latin1')
        cls.result = np.load(request.urlretrieve(os.path.join(
            base_url,
            'eval_instance_segmentation_coco_result_2019_02_12.npz'))[0],
            encoding='latin1')

    def test_eval_instance_segmentation_coco(self):
        pred_masks = self.result['masks']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        gt_masks = self.dataset['masks']
        gt_labels = self.dataset['labels']
        gt_crowdeds = self.dataset['crowdeds']
        gt_areas = self.dataset['areas']

        result = eval_instance_segmentation_coco(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, gt_areas, gt_crowdeds)

        expected = {
            'map/iou=0.50:0.95/area=all/max_dets=100': 0.32170935,
            'map/iou=0.50/area=all/max_dets=100': 0.56469292,
            'map/iou=0.75/area=all/max_dets=100': 0.30133106,
            'map/iou=0.50:0.95/area=small/max_dets=100': 0.38737403,
            'map/iou=0.50:0.95/area=medium/max_dets=100': 0.31018272,
            'map/iou=0.50:0.95/area=large/max_dets=100': 0.32693391,
            'mar/iou=0.50:0.95/area=all/max_dets=1': 0.27037258,
            'mar/iou=0.50:0.95/area=all/max_dets=10': 0.41759154,
            'mar/iou=0.50:0.95/area=all/max_dets=100': 0.41898236,
            'mar/iou=0.50:0.95/area=small/max_dets=100': 0.46944986,
            'mar/iou=0.50:0.95/area=medium/max_dets=100': 0.37675923,
            'mar/iou=0.50:0.95/area=large/max_dets=100': 0.38147151
        }

        non_existent_labels = np.setdiff1d(
            np.arange(max(result['existent_labels'])),
            result['existent_labels'])
        for key, item in expected.items():
            non_mean_key = key[1:]
            self.assertIsInstance(result[non_mean_key], np.ndarray)
            self.assertEqual(result[non_mean_key].shape, (80,))
            self.assertTrue(
                np.all(np.isnan(result[non_mean_key][non_existent_labels])))
            np.testing.assert_almost_equal(
                result[key], expected[key], decimal=5)


testing.run_module(__name__, __file__)
