from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.fpn import RPN
from chainercv.links.model.fpn import rpn_loss


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class TestRPN(unittest.TestCase):

    def setUp(self):
        self.link = RPN(scales=(1 / 2, 1 / 4, 1 / 8))

    def _check_call(self):
        hs = [
            chainer.Variable(_random_array(self.link.xp, (2, 64, 32, 32))),
            chainer.Variable(_random_array(self.link.xp, (2, 64, 16, 16))),
            chainer.Variable(_random_array(self.link.xp, (2, 64, 8, 8))),
        ]

        locs, confs = self.link(hs)

        self.assertEqual(len(locs), 3)
        self.assertEqual(len(confs), 3)
        for l in range(3):
            self.assertIsInstance(locs[l], chainer.Variable)
            self.assertIsInstance(locs[l].array, self.link.xp.ndarray)
            self.assertEqual(locs[l].shape, (2, (32 * 32 >> 2 * l) * 3, 4))

            self.assertIsInstance(confs[l], chainer.Variable)
            self.assertIsInstance(confs[l].array, self.link.xp.ndarray)
            self.assertEqual(confs[l].shape, (2, (32 * 32 >> 2 * l) * 3))

    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def _check_anchors(self):
        anchors = self.link.anchors(((32, 32), (16, 16), (8, 8)))

        self.assertEqual(len(anchors), 3)
        for l in range(3):
            self.assertIsInstance(anchors[l], self.link.xp.ndarray)
            self.assertEqual(anchors[l].shape, ((32 * 32 >> 2 * l) * 3, 4))

    def test_anchors_cpu(self):
        self._check_anchors()

    @attr.gpu
    def test_anchors_gpu(self):
        self.link.to_gpu()
        self._check_anchors()

    def _check_decode(self):
        locs = [
            chainer.Variable(_random_array(
                self.link.xp, (2, 32 * 32 * 3, 4))),
            chainer.Variable(_random_array(
                self.link.xp, (2, 16 * 16 * 3, 4))),
            chainer.Variable(_random_array(
                self.link.xp, (2, 8 * 8 * 3, 4))),
        ]
        confs = [
            chainer.Variable(_random_array(
                self.link.xp, (2, 32 * 32 * 3))),
            chainer.Variable(_random_array(
                self.link.xp, (2, 16 * 16 * 3))),
            chainer.Variable(_random_array(
                self.link.xp, (2, 8 * 8 * 3))),
        ]
        anchors = self.link.anchors(((32, 32), (16, 16), (8, 8)))

        rois, roi_indices = self.link.decode(
            locs, confs, anchors, (2, 3, 64, 64))

        self.assertIsInstance(rois, self.link.xp.ndarray)
        self.assertIsInstance(roi_indices, self.link.xp.ndarray)
        self.assertEqual(rois.shape[0], roi_indices.shape[0])
        self.assertEqual(rois.shape[1:], (4,))
        self.assertEqual(roi_indices.shape[1:], ())

    def test_decode_cpu(self):
        self._check_decode()

    @attr.gpu
    def test_decode_gpu(self):
        self.link.to_gpu()
        self._check_decode()


class TestRPNLoss(unittest.TestCase):

    def _check_rpn_loss(self, xp):
        locs = [
            chainer.Variable(_random_array(
                xp, (2, 32 * 32 * 3, 4))),
            chainer.Variable(_random_array(
                xp, (2, 16 * 16 * 3, 4))),
            chainer.Variable(_random_array(
                xp, (2, 8 * 8 * 3, 4))),
        ]
        confs = [
            chainer.Variable(_random_array(
                xp, (2, 32 * 32 * 3))),
            chainer.Variable(_random_array(
                xp, (2, 16 * 16 * 3))),
            chainer.Variable(_random_array(
                xp, (2, 8 * 8 * 3))),
        ]
        anchors = RPN(scales=(1 / 2, 1 / 4, 1 / 8)) \
            .anchors(((32, 32), (16, 16), (8, 8)))
        bboxes = [
            xp.array(((2, 4, 6, 7), (1, 12, 3, 30)), dtype=np.float32),
            xp.array(((10, 2, 12, 12),), dtype=np.float32),
        ]

        loc_loss, conf_loss = rpn_loss(
            locs, confs, anchors, ((480, 640), (320, 320)), bboxes)

        self.assertIsInstance(loc_loss, chainer.Variable)
        self.assertIsInstance(loc_loss.array, xp.ndarray)
        self.assertEqual(loc_loss.shape, ())

        self.assertIsInstance(conf_loss, chainer.Variable)
        self.assertIsInstance(conf_loss.array, xp.ndarray)
        self.assertEqual(conf_loss.shape, ())

    def test_rpn_loss_cpu(self):
        self._check_rpn_loss(np)

    @attr.gpu
    def test_rpn_loss_gpu(self):
        import cupy
        self._check_rpn_loss(cupy)


testing.run_module(__name__, __file__)
