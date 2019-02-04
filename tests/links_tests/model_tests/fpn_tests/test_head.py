from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.fpn import Head


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


@testing.parameterize(
    {'n_class': 1 + 1},
    {'n_class': 5 + 1},
    {'n_class': 20 + 1},
)
class TestHead(unittest.TestCase):

    def setUp(self):
        self.link = Head(n_class=self.n_class, scales=(1 / 2, 1 / 4, 1 / 8))

    def _check_call(self):
        hs = [
            chainer.Variable(_random_array(self.link.xp, (2, 64, 32, 32))),
            chainer.Variable(_random_array(self.link.xp, (2, 64, 16, 16))),
            chainer.Variable(_random_array(self.link.xp, (2, 64, 8, 8))),
        ]
        rois = [
            self.link.xp.array(((4, 1, 6, 3),), dtype=np.float32),
            self.link.xp.array(
                ((0, 1, 2, 3), (5, 4, 10, 6)), dtype=np.float32),
            self.link.xp.array(((10, 4, 12, 10),), dtype=np.float32),
        ]
        roi_indices = [
            self.link.xp.array((0,), dtype=np.int32),
            self.link.xp.array((1, 0), dtype=np.int32),
            self.link.xp.array((1,), dtype=np.int32),
        ]

        locs, confs = self.link(hs, rois, roi_indices)

        self.assertIsInstance(locs, chainer.Variable)
        self.assertIsInstance(locs.array, self.link.xp.ndarray)
        self.assertEqual(locs.shape, (4, self.n_class, 4))

        self.assertIsInstance(confs, chainer.Variable)
        self.assertIsInstance(confs.array, self.link.xp.ndarray)
        self.assertEqual(confs.shape, (4, self.n_class))

    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def _check_distribute(self):
        rois = self.link.xp.array((
            (0, 0, 10, 10),
            (0, 1000, 0, 1000),
            (0, 0, 224, 224),
            (100, 100, 224, 224),
        ), dtype=np.float32)
        roi_indices = self.link.xp.array((0, 1, 0, 0), dtype=np.int32)

        rois, roi_indices = self.link.distribute(rois, roi_indices)

        self.assertEqual(len(rois), 3)
        self.assertEqual(len(roi_indices), 3)
        for l in range(3):
            self.assertIsInstance(rois[l], self.link.xp.ndarray)
            self.assertIsInstance(roi_indices[l], self.link.xp.ndarray)

            self.assertEqual(rois[l].shape[0], roi_indices[l].shape[0])
            self.assertEqual(rois[l].shape[1:], (4,))
            self.assertEqual(roi_indices[l].shape[1:], ())

        self.assertEqual(sum(rois[l].shape[0] for l in range(3)), 4)

    def test_distribute_cpu(self):
        self._check_distribute()

    @attr.gpu
    def test_distribute_gpu(self):
        self.link.to_gpu()
        self._check_distribute()

    def _check_decode(self):
        rois = [
            self.link.xp.array(((4, 1, 6, 3),), dtype=np.float32),
            self.link.xp.array(
                ((0, 1, 2, 3), (5, 4, 10, 6)), dtype=np.float32),
            self.link.xp.array(((10, 4, 12, 10),), dtype=np.float32),
        ]
        roi_indices = [
            self.link.xp.array((0,), dtype=np.int32),
            self.link.xp.array((1, 0), dtype=np.int32),
            self.link.xp.array((1,), dtype=np.int32),
        ]
        locs = chainer.Variable(_random_array(
            self.link.xp, (4, self.n_class, 4)))
        confs = chainer.Variable(_random_array(
            self.link.xp, (4, self.n_class)))

        bboxes, labels, scores = self.link.decode(
            rois, roi_indices,
            (0.4, 0.2), ((100, 100), (200, 200)),
            locs, confs, 0.5, 0.1)

        self.assertEqual(len(bboxes), 2)
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(scores), 2)
        for n in range(2):
            self.assertIsInstance(bboxes[n], self.link.xp.ndarray)
            self.assertIsInstance(labels[n], self.link.xp.ndarray)
            self.assertIsInstance(scores[n], self.link.xp.ndarray)

            self.assertEqual(bboxes[n].shape[0], labels[n].shape[0])
            self.assertEqual(bboxes[n].shape[0], scores[n].shape[0])
            self.assertEqual(bboxes[n].shape[1:], (4,))
            self.assertEqual(labels[n].shape[1:], (1,))
            self.assertEqual(scores[n].shape[1:], (1,))


testing.run_module(__name__, __file__)
