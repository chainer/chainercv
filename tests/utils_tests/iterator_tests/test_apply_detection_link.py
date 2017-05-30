import numpy as np
import unittest

import chainer
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.utils import apply_detection_link


class DummyDetectionLink(chainer.Link):
    n_fg_class = 20

    def predict(self, imgs):
        bboxes = list()
        labels = list()
        scores = list()

        for _ in imgs:
            n_bbox = np.random.randint(0, 20)
            bboxes.append(np.random.uniform(size=(n_bbox, 4)))
            labels.append(np.random.randint(0, self.n_fg_class, size=n_bbox))
            scores.append(np.random.uniform(size=n_bbox))

        return bboxes, labels, scores


@testing.parameterize(
    {'with_hook': False},
    {'with_hook': True},
)
class TestApplyDetectionLink(unittest.TestCase):

    def setUp(self):
        self.link = DummyDetectionLink()

        self.imgs = list()
        for _ in range(5):
            H, W = np.random.randint(8, 16, size=2)
            self.imgs.append(np.random.randint(0, 256, size=(3, H, W)))

    def test_image_dataset(self):
        dataset = self.imgs
        iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)

        if self.with_hook:
            def hook(
                    pred_bboxes, pred_labels, pred_scores, gt_values):
                self.assertEqual(len(pred_labels), len(pred_bboxes))
                self.assertEqual(len(pred_scores), len(pred_bboxes))
                self.assertEqual(len(gt_values), 0)
        else:
            hook = None

        pred_bboxes, pred_labels, pred_scores, gt_values = \
            apply_detection_link(self.link, iterator, hook=hook)

        self.assertEqual(len(list(pred_bboxes)), len(dataset))
        self.assertEqual(len(list(pred_labels)), len(dataset))
        self.assertEqual(len(list(pred_scores)), len(dataset))

        self.assertEqual(len(gt_values), 0)

    def test_general_dataset(self):
        strs = ['a', 'bc', 'def', 'ghij', 'klmno']
        nums = [0, 1, 2, 3, 4]
        arrays = [np.random.uniform(size=10) for _ in range(5)]
        dataset = chainer.datasets.TupleDataset(
            self.imgs, strs, nums, arrays)
        iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)

        if self.with_hook:
            def hook(
                    pred_bboxes, pred_labels, pred_scores, gt_values):
                self.assertEqual(len(pred_labels), len(pred_bboxes))
                self.assertEqual(len(pred_scores), len(pred_bboxes))
                self.assertEqual(len(gt_values), 3)
                self.assertEqual(len(gt_values[0]), len(pred_bboxes))
                self.assertEqual(len(gt_values[1]), len(pred_bboxes))
                self.assertEqual(len(gt_values[2]), len(pred_bboxes))
        else:
            hook = None

        pred_bboxes, pred_labels, pred_scores, gt_values = \
            apply_detection_link(self.link, iterator, hook=hook)

        self.assertEqual(len(list(pred_bboxes)), len(dataset))
        self.assertEqual(len(list(pred_labels)), len(dataset))
        self.assertEqual(len(list(pred_scores)), len(dataset))

        self.assertEqual(len(gt_values), 3)
        self.assertEqual(list(gt_values[0]), strs)
        self.assertEqual(list(gt_values[1]), nums)
        self.assertEqual(list(gt_values[2]), arrays)


testing.run_module(__name__, __file__)
