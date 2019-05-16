from __future__ import division

import numpy as np
import unittest

import chainer
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.utils.testing import attr

from chainermn import create_communicator


class _SemanticSegmentationStubLink(chainer.Link):

    def __init__(self, labels, initial_count):
        super(_SemanticSegmentationStubLink, self).__init__()
        self.count = initial_count
        self.labels = labels

    def predict(self, imgs):
        n_img = len(imgs)
        labels = self.labels[self.count:self.count + n_img]

        self.count += n_img
        return labels


class TestSemanticSegmentationEvaluator(unittest.TestCase):

    def _set_up(self, comm):
        batchsize_per_process = 1
        batchsize = (batchsize_per_process * comm.size
                     if comm is not None else batchsize_per_process)
        self.label_names = ('a', 'b', 'c')
        imgs = [np.random.uniform(size=(3, 2, 3)) for _ in range(2)]
        # There are labels for 'a' and 'b', but none for 'c'.
        pred_labels = [
            np.array([[1, 1, 1], [0, 0, 1]]),
            np.array([[1, 1, 1], [0, 0, 1]])]
        gt_labels = [
            np.array([[1, 0, 0], [0, -1, 1]]),
            np.array([[1, 0, 0], [0, -1, 1]])]

        self.iou_a = 1 / 3
        self.iou_b = 2 / 4
        self.pixel_accuracy = 3 / 5
        self.class_accuracy_a = 1 / 3
        self.class_accuracy_b = 2 / 2
        self.miou = np.mean((self.iou_a, self.iou_b))
        self.mean_class_accuracy = np.mean(
            (self.class_accuracy_a, self.class_accuracy_b))

        self.dataset = TupleDataset(imgs, gt_labels)
        if comm is None or comm.rank == 0:
            initial_count = 0
            iterator = SerialIterator(
                self.dataset, batchsize, repeat=False, shuffle=False)
        else:
            initial_count = comm.rank * batchsize_per_process
            iterator = None
        self.link = _SemanticSegmentationStubLink(pred_labels, initial_count)
        self.evaluator = SemanticSegmentationEvaluator(
            iterator, self.link, self.label_names, comm=comm)

    def _check_evaluate(self, comm=None):
        self._set_up(comm)
        reporter = chainer.Reporter()
        reporter.add_observer('main', self.link)
        with reporter:
            eval_ = self.evaluator.evaluate()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(eval_, {})
            return

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

    def test_evaluate(self):
        self._check_evaluate()

    @attr.mpi
    def test_evaluate_with_comm(self):
        comm = create_communicator('naive')
        self._check_evaluate(comm)

    def _check_call(self, comm=None):
        self._set_up(comm)
        eval_ = self.evaluator()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(eval_, {})
            return
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

    def test_call(self):
        self._check_call()

    @attr.mpi
    def test_call_with_comm(self):
        comm = create_communicator('naive')
        self._check_call(comm)

    def _check_evaluator_name(self, comm=None):
        self._set_up(comm)
        self.evaluator.name = 'eval'
        eval_ = self.evaluator()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(eval_, {})
            return
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

    def test_evaluator_name(self):
        self._check_evaluator_name()

    @attr.mpi
    def test_evaluator_name_with_comm(self):
        comm = create_communicator('naive')
        self._check_evaluator_name(comm)

    def _check_current_report(self, comm=None):
        self._set_up(comm)
        reporter = chainer.Reporter()
        with reporter:
            eval_ = self.evaluator()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(eval_, {})
            return
        # The result is reported to the current reporter.
        np.testing.assert_equal(reporter.observation, eval_)

    def test_current_report(self):
        self._check_current_report()

    @attr.mpi
    def test_current_report_with_comm(self):
        comm = create_communicator('naive')
        self._check_current_report(comm)


testing.run_module(__name__, __file__)
