import numpy as np
import unittest

import chainer
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.extensions import InstanceSegmentationVOCEvaluator
from chainercv.utils.testing import attr

from chainermn import create_communicator


class _InstanceSegmentationStubLink(chainer.Link):

    def __init__(self, masks, labels, initial_count):
        super(_InstanceSegmentationStubLink, self).__init__()
        self.count = initial_count
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

    def _set_up(self, comm):
        batchsize_per_process = 5
        batchsize = (batchsize_per_process * comm.size
                     if comm is not None else batchsize_per_process)
        if comm is None or comm.rank == 0:
            masks = [np.random.uniform(size=(5, 32, 48)) > 0.5
                     for _ in range(10)]
            labels = [np.ones((5,), dtype=np.int32) for _ in range(10)]

            dataset = TupleDataset(
                np.random.uniform(size=(10, 3, 32, 48)),
                masks, labels)
            initial_count = 0
            iterator = SerialIterator(
                dataset, batchsize, repeat=False, shuffle=False)
        else:
            masks = None
            labels = None
            initial_count = comm.rank * batchsize_per_process
            iterator = None

        if comm is not None:
            masks = comm.bcast_obj(masks)
            labels = comm.bcast_obj(labels)

        self.link = _InstanceSegmentationStubLink(masks, labels, initial_count)
        self.evaluator = InstanceSegmentationVOCEvaluator(
            iterator, self.link, label_names=('cls0', 'cls1', 'cls2'),
            comm=comm)
        self.expected_ap = 1

    def _check_evaluate(self, comm=None):
        self._set_up(comm)
        reporter = chainer.Reporter()
        reporter.add_observer('target', self.link)
        with reporter:
            mean = self.evaluator.evaluate()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(mean, {})
            return

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        self.assertEqual(len(reporter.observation), 0)

        np.testing.assert_equal(mean['target/map'], self.expected_ap)
        np.testing.assert_equal(mean['target/ap/cls0'], np.nan)
        np.testing.assert_equal(mean['target/ap/cls1'], self.expected_ap)
        np.testing.assert_equal(mean['target/ap/cls2'], np.nan)

    def test_evaluate(self):
        self._check_evaluate()

    @attr.mpi
    def test_evaluate_with_comm(self):
        comm = create_communicator('naive')
        self._check_evaluate(comm)

    def _check_call(self, comm=None):
        self._set_up(comm)
        mean = self.evaluator()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(mean, {})
            return
        # main is used as default
        np.testing.assert_equal(mean['main/map'], self.expected_ap)
        np.testing.assert_equal(mean['main/ap/cls0'], np.nan)
        np.testing.assert_equal(mean['main/ap/cls1'], self.expected_ap)
        np.testing.assert_equal(mean['main/ap/cls2'], np.nan)

    def test_call(self):
        self._check_call()

    @attr.mpi
    def test_call_with_comm(self):
        comm = create_communicator('naive')
        self._check_call(comm)

    def _check_evaluator_name(self, comm=None):
        self._set_up(comm)
        self.evaluator.name = 'eval'
        mean = self.evaluator()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(mean, {})
            return
        # name is used as a prefix
        np.testing.assert_equal(mean['eval/main/map'], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/ap/cls0'], np.nan)
        np.testing.assert_equal(mean['eval/main/ap/cls1'], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/ap/cls2'], np.nan)

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
            mean = self.evaluator()
        if comm is not None and not comm.rank == 0:
            self.assertEqual(mean, {})
            return
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, mean)

    def test_current_report(self):
        self._check_current_report()

    @attr.mpi
    def test_current_report_with_comm(self):
        comm = create_communicator('naive')
        self._check_current_report(comm)


testing.run_module(__name__, __file__)
