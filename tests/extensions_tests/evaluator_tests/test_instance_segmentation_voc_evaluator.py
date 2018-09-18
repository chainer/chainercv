import numpy as np
import unittest

import chainer
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.extensions import InstanceSegmentationVOCEvaluator


class _InstanceSegmentationStubLink(chainer.Link):

    def __init__(self, masks, labels):
        super(_InstanceSegmentationStubLink, self).__init__()
        self.count = 0
        self.masks = masks
        self.labels = labels

    def predict(self, imgs):
        n_img = len(imgs)
        masks = self.masks[self.count:self.count + n_img]
        labels = self.labels[self.count:self.count + n_img]
        scores = [np.ones_like(l) for l in labels]

        self.count += n_img

        return masks, labels, scores


class TestInstanceSegmentationVOCEvaluator(unittest.TestCase):

    def setUp(self):
        masks = np.random.uniform(size=(10, 5, 32, 48)) > 0.5
        labels = np.ones((10, 5), dtype=np.int32)
        self.dataset = TupleDataset(
            np.random.uniform(size=(10, 3, 32, 48)),
            masks, labels)
        self.link = _InstanceSegmentationStubLink(masks, labels)
        self.iterator = SerialIterator(
            self.dataset, 1, repeat=False, shuffle=False)
        self.evaluator = InstanceSegmentationVOCEvaluator(
            self.iterator, self.link, label_names=('cls0', 'cls1', 'cls2'))
        self.expected_ap = 1

    def test_evaluate(self):
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.link)
        with reporter:
            mean = self.evaluator.evaluate()

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        self.assertEqual(len(reporter.observation), 0)

        np.testing.assert_equal(mean['target/map'], self.expected_ap)
        np.testing.assert_equal(mean['target/ap/cls0'], np.nan)
        np.testing.assert_equal(mean['target/ap/cls1'], self.expected_ap)
        np.testing.assert_equal(mean['target/ap/cls2'], np.nan)

    def test_call(self):
        mean = self.evaluator()
        # main is used as default
        np.testing.assert_equal(mean['main/map'], self.expected_ap)
        np.testing.assert_equal(mean['main/ap/cls0'], np.nan)
        np.testing.assert_equal(mean['main/ap/cls1'], self.expected_ap)
        np.testing.assert_equal(mean['main/ap/cls2'], np.nan)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        mean = self.evaluator()
        # name is used as a prefix
        np.testing.assert_equal(mean['eval/main/map'], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/ap/cls0'], np.nan)
        np.testing.assert_equal(mean['eval/main/ap/cls1'], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/ap/cls2'], np.nan)

    def test_current_report(self):
        reporter = chainer.Reporter()
        with reporter:
            mean = self.evaluator()
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, mean)


testing.run_module(__name__, __file__)
