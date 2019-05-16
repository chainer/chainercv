import numpy as np
import unittest

import chainer
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer import testing

from chainercv.extensions import InstanceSegmentationCOCOEvaluator
from chainercv.utils.testing import attr

from chainermn import create_communicator

try:
    import pycocotools.coco  # NOQA
    _available = True
except ImportError:
    _available = False


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


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestInstanceSegmentationCOCOEvaluator(unittest.TestCase):

    def _set_up(self, comm):
        if comm is None or comm.rank == 0:
            masks = np.random.uniform(size=(10, 5, 32, 48)) > 0.5
            labels = np.ones((10, 5), dtype=np.int32)
            dataset = TupleDataset(
                np.random.uniform(size=(10, 3, 32, 48)),
                masks, labels)
            initial_count = 0
            iterator = SerialIterator(
                dataset, 1, repeat=False, shuffle=False)
        else:
            masks = None
            labels = None
            initial_count = comm.rank * 5
            iterator = None
        self.link = _InstanceSegmentationStubLink(masks, labels, initial_count)
        self.evaluator = InstanceSegmentationCOCOEvaluator(
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

        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        np.testing.assert_equal(
            mean['target/m{}'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['target/{}/cls0'.format(key)], np.nan)
        np.testing.assert_equal(
            mean['target/{}/cls1'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['target/{}/cls2'.format(key)], np.nan)

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
        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        np.testing.assert_equal(mean['main/m{}'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['main/{}/cls0'.format(key)], np.nan)
        np.testing.assert_equal(
            mean['main/{}/cls1'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['main/{}/cls2'.format(key)], np.nan)

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
        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        np.testing.assert_equal(
            mean['eval/main/m{}'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/{}/cls0'.format(key)], np.nan)
        np.testing.assert_equal(
            mean['eval/main/{}/cls1'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/{}/cls2'.format(key)], np.nan)

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
