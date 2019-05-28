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

    def __init__(self, masks, labels, initial_count=0):
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

    def setUp(self):
        masks = np.random.uniform(size=(10, 5, 32, 48)) > 0.5
        labels = np.ones((10, 5), dtype=np.int32)
        self.dataset = TupleDataset(
            np.random.uniform(size=(10, 3, 32, 48)),
            masks, labels)
        self.link = _InstanceSegmentationStubLink(masks, labels)
        self.iterator = SerialIterator(
            self.dataset, 1, repeat=False, shuffle=False)
        self.evaluator = InstanceSegmentationCOCOEvaluator(
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

        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        np.testing.assert_equal(
            mean['target/m{}'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['target/{}/cls0'.format(key)], np.nan)
        np.testing.assert_equal(
            mean['target/{}/cls1'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['target/{}/cls2'.format(key)], np.nan)

    def test_call(self):
        mean = self.evaluator()
        # main is used as default
        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        np.testing.assert_equal(mean['main/m{}'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['main/{}/cls0'.format(key)], np.nan)
        np.testing.assert_equal(
            mean['main/{}/cls1'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['main/{}/cls2'.format(key)], np.nan)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        mean = self.evaluator()
        # name is used as a prefix
        key = 'ap/iou=0.50:0.95/area=all/max_dets=100'
        np.testing.assert_equal(
            mean['eval/main/m{}'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/{}/cls0'.format(key)], np.nan)
        np.testing.assert_equal(
            mean['eval/main/{}/cls1'.format(key)], self.expected_ap)
        np.testing.assert_equal(mean['eval/main/{}/cls2'.format(key)], np.nan)

    def test_current_report(self):
        reporter = chainer.Reporter()
        with reporter:
            mean = self.evaluator()
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, mean)


@unittest.skipUnless(_available, 'pycocotools is not installed')
class TestInstanceSegmentationCOCOEvaluatorMPI(unittest.TestCase):

    def setUp(self):
        comm = create_communicator('naive')
        self.comm = comm

        batchsize_per_process = 5
        batchsize = batchsize_per_process * comm.size
        if comm.rank == 0:
            masks = [np.random.uniform(size=(5, 32, 48)) > 0.5
                     for _ in range(10)]
            labels = [np.ones((5,), dtype=np.int32)
                      for _ in range(10)]
        else:
            masks = None
            labels = None
        initial_count = comm.rank * batchsize_per_process

        masks = comm.bcast_obj(masks)
        labels = comm.bcast_obj(labels)
        self.masks = masks
        self.labels = labels

        self.dataset = TupleDataset(
            np.random.uniform(size=(10, 3, 32, 48)),
            masks, labels)
        self.initial_count = initial_count
        self.batchsize = batchsize

    @attr.mpi
    def test_consistency(self):
        reporter = chainer.Reporter()

        if self.comm.rank == 0:
            multi_iterator = SerialIterator(
                self.dataset, self.batchsize, repeat=False, shuffle=False)
        else:
            multi_iterator = None
        multi_link = _InstanceSegmentationStubLink(
            self.masks, self.labels, self.initial_count)
        multi_evaluator = InstanceSegmentationCOCOEvaluator(
            multi_iterator, multi_link,
            label_names=('cls0', 'cls1', 'cls2'),
            comm=self.comm)
        reporter.add_observer('target', multi_link)
        with reporter:
            multi_mean = multi_evaluator.evaluate()

        if self.comm.rank != 0:
            self.assertEqual(multi_mean, {})
            return

        single_iterator = SerialIterator(
            self.dataset, self.batchsize, repeat=False, shuffle=False)
        single_link = _InstanceSegmentationStubLink(
            self.masks, self.labels)
        single_evaluator = InstanceSegmentationCOCOEvaluator(
            single_iterator, single_link,
            label_names=('cls0', 'cls1', 'cls2'))
        reporter.add_observer('target', single_link)
        with reporter:
            single_mean = single_evaluator.evaluate()

        self.assertEqual(set(multi_mean.keys()), set(single_mean.keys()))
        for key in multi_mean.keys():
            np.testing.assert_equal(single_mean[key], multi_mean[key])


testing.run_module(__name__, __file__)
