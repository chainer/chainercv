import numpy as np
import unittest

import chainer
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.extensions import DetectionVOCAPEvaluator
from chainercv.utils import generate_random_bbox


class _DetectionStubLink(chainer.Link):

    def __init__(self, bboxes, labels):
        super(_DetectionStubLink, self).__init__()
        self.count = 0
        self.bboxes = bboxes
        self.labels = labels

    def predict(self, imgs):
        n_img = len(imgs)
        bboxes = self.bboxes[self.count:self.count + n_img]
        labels = self.labels[self.count:self.count + n_img]
        scores = [np.ones_like(l) for l in labels]

        self.count += n_img

        return bboxes, labels, scores


class TestDetectionVOCAPEvaluator(unittest.TestCase):

    def setUp(self):
        bboxes = [generate_random_bbox(5, (256, 324), 24, 120)
                  for _ in range(10)]
        labels = np.ones((10, 5))
        self.dataset = TupleDataset(
            np.random.uniform(size=(10, 3, 32, 48)),
            bboxes,
            labels)
        self.link = _DetectionStubLink(bboxes, labels)
        self.iterator = SerialIterator(
            self.dataset, 5, repeat=False, shuffle=False)
        self.evaluator = DetectionVOCAPEvaluator(self.iterator, self.link)
        self.expect_map = 1

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.link)
        with reporter:
            mean = self.evaluator.evaluate()

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        self.assertEqual(len(reporter.observation), 0)

        self.assertEqual(mean['target/map'], self.expect_map)

    def test_call(self):
        mean = self.evaluator()
        # main is used as default
        self.assertEqual(mean['main/map'], self.expect_map)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        mean = self.evaluator()
        # name is used as a prefix
        self.assertAlmostEqual(
            mean['eval/main/map'], self.expect_map)

    def test_current_report(self):
        reporter = chainer.Reporter()
        with reporter:
            mean = self.evaluator()
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, mean)


testing.run_module(__name__, __file__)
