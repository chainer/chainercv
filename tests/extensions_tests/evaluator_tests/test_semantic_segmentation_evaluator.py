from __future__ import division

import numpy as np
import unittest

import chainer
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.extensions import SemanticSegmentationEvaluator


class _SemanticSegmentationStubLink(chainer.Link):

    def __init__(self, labels):
        super(_SemanticSegmentationStubLink, self).__init__()
        self.count = 0
        self.labels = labels

    def predict(self, imgs):
        n_img = len(imgs)
        labels = self.labels[self.count:self.count + n_img]

        self.count += n_img
        return labels


class TestSemanticSegmentationEvaluator(unittest.TestCase):

    def setUp(self):
        self.label_names = ('a', 'b', 'c')
        imgs = np.random.uniform(size=(1, 3, 2, 3))
        # There are labels for 'a' and 'b', but none for 'c'.
        pred_labels = np.array([[[1, 1, 1], [0, 0, 1]]])
        gt_labels = np.array([[[1, 0, 0], [0, -1, 1]]])

        self.iou_a = 1 / 3
        self.iou_b = 2 / 4
        self.pixel_accuracy = 3 / 5
        self.class_accuracy_a = 1 / 3
        self.class_accuracy_b = 2 / 2
        self.miou = np.mean((self.iou_a, self.iou_b))
        self.mean_class_accuracy = np.mean(
            (self.class_accuracy_a, self.class_accuracy_b))

        self.dataset = TupleDataset(imgs, gt_labels)
        self.link = _SemanticSegmentationStubLink(pred_labels)
        self.iterator = SerialIterator(
            self.dataset, 5, repeat=False, shuffle=False)
        self.evaluator = SemanticSegmentationEvaluator(
            self.iterator, self.link, self.label_names)

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('main', self.link)
        with reporter:
            eval_ = self.evaluator.evaluate()

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        np.testing.assert_equal(len(reporter.observation), 0)

        np.testing.assert_equal(eval_['main/miou'], self.miou)
        np.testing.assert_equal(eval_['main/pixel_accuracy'],
                                self.pixel_accuracy)
        np.testing.assert_equal(eval_['main/mean_class_accuracy'],
                                self.mean_class_accuracy)
        np.testing.assert_equal(eval_['main/iou/a'], self.iou_a)
        np.testing.assert_equal(eval_['main/iou/b'], self.iou_b)
        np.testing.assert_equal(eval_['main/iou/c'], np.nan)
        np.testing.assert_equal(eval_['main/class_accuracy/a'],
                                self.class_accuracy_a)
        np.testing.assert_equal(eval_['main/class_accuracy/b'],
                                self.class_accuracy_b)
        np.testing.assert_equal(eval_['main/class_accuracy/c'], np.nan)

    def test_call(self):
        eval_ = self.evaluator()
        # main is used as default
        np.testing.assert_equal(eval_['main/miou'], self.miou)
        np.testing.assert_equal(eval_['main/pixel_accuracy'],
                                self.pixel_accuracy)
        np.testing.assert_equal(eval_['main/mean_class_accuracy'],
                                self.mean_class_accuracy)
        np.testing.assert_equal(eval_['main/iou/a'], self.iou_a)
        np.testing.assert_equal(eval_['main/iou/b'], self.iou_b)
        np.testing.assert_equal(eval_['main/iou/c'], np.nan)
        np.testing.assert_equal(eval_['main/class_accuracy/a'],
                                self.class_accuracy_a)
        np.testing.assert_equal(eval_['main/class_accuracy/b'],
                                self.class_accuracy_b)
        np.testing.assert_equal(eval_['main/class_accuracy/c'], np.nan)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        eval_ = self.evaluator()
        # name is used as a prefix
        np.testing.assert_equal(eval_['eval/main/miou'], self.miou)
        np.testing.assert_equal(eval_['eval/main/pixel_accuracy'],
                                self.pixel_accuracy)
        np.testing.assert_equal(eval_['eval/main/mean_class_accuracy'],
                                self.mean_class_accuracy)
        np.testing.assert_equal(eval_['eval/main/iou/a'], self.iou_a)
        np.testing.assert_equal(eval_['eval/main/iou/b'], self.iou_b)
        np.testing.assert_equal(eval_['eval/main/iou/c'], np.nan)
        np.testing.assert_equal(eval_['eval/main/class_accuracy/a'],
                                self.class_accuracy_a)
        np.testing.assert_equal(eval_['eval/main/class_accuracy/b'],
                                self.class_accuracy_b)
        np.testing.assert_equal(eval_['eval/main/class_accuracy/c'], np.nan)

    def test_current_report(self):
        reporter = chainer.Reporter()
        with reporter:
            eval_ = self.evaluator()
        # The result is reported to the current reporter.
        np.testing.assert_equal(reporter.observation, eval_)


testing.run_module(__name__, __file__)
