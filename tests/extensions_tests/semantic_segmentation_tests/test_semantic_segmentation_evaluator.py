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
        n_class = 3
        self.label_names = ('a', 'b', 'c')
        imgs = np.random.uniform(size=(10, 3, 5, 5))
        labels = np.random.randint(
            low=0, high=n_class, size=(10, 5, 5), dtype=np.int32)
        self.dataset = TupleDataset(imgs, labels)
        self.link = _SemanticSegmentationStubLink(labels)
        self.iterator = SerialIterator(
            self.dataset, 5, repeat=False, shuffle=False)
        self.evaluator = SemanticSegmentationEvaluator(
            self.iterator, self.link, n_class, self.label_names)

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.link)
        with reporter:
            eval_ = self.evaluator.evaluate()

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        self.assertEqual(len(reporter.observation), 0)

        self.assertEqual(eval_['target/miou'], 1.)
        for label_name in self.label_names:
            self.assertEqual(eval_['target/{}/iou'.format(label_name)], 1)

    def test_call(self):
        eval_ = self.evaluator()
        # main is used as default
        self.assertEqual(eval_['main/miou'], 1)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        eval_ = self.evaluator()
        # name is used as a prefix
        self.assertAlmostEqual(
            eval_['eval/main/miou'], 1)

    def test_current_report(self):
        reporter = chainer.Reporter()
        with reporter:
            eval_ = self.evaluator()
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, eval_)


testing.run_module(__name__, __file__)
