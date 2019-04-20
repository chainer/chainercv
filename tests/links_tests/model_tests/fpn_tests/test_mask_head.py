from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.fpn import mask_head_loss_post
from chainercv.links.model.fpn import mask_head_loss_pre
from chainercv.links.model.fpn import MaskHead

from chainercv.utils import mask_to_bbox

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


@testing.parameterize(
    {'n_class': 1 + 1},
    {'n_class': 5 + 1},
    {'n_class': 20 + 1},
)
class TestMaskHead(unittest.TestCase):

    def setUp(self):
        self.link = MaskHead(
            n_class=self.n_class, scales=(1 / 2, 1 / 4, 1 / 8))

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

        segs = self.link(hs, rois, roi_indices)

        self.assertIsInstance(segs, chainer.Variable)
        self.assertIsInstance(segs.array, self.link.xp.ndarray)
        self.assertEqual(
            segs.shape,
            (4, self.n_class, self.link.segm_size, self.link.segm_size))

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
        n_roi = len(roi_indices)

        rois, roi_indices, order = self.link.distribute(rois, roi_indices)

        self.assertEqual(len(rois), 3)
        self.assertEqual(len(roi_indices), 3)
        for l in range(3):
            self.assertIsInstance(rois[l], self.link.xp.ndarray)
            self.assertIsInstance(roi_indices[l], self.link.xp.ndarray)

            self.assertEqual(rois[l].shape[0], roi_indices[l].shape[0])
            self.assertEqual(rois[l].shape[1:], (4,))
            self.assertEqual(roi_indices[l].shape[1:], ())

        self.assertEqual(sum(rois[l].shape[0] for l in range(3)), 4)

        self.assertEqual(len(order), n_roi)
        self.assertIsInstance(order, self.link.xp.ndarray)

    def test_distribute_cpu(self):
        self._check_distribute()

    @attr.gpu
    def test_distribute_gpu(self):
        self.link.to_gpu()
        self._check_distribute()

    def _check_decode(self):
        segms = [
            _random_array(
                self.link.xp,
                (1, self.n_class, self.link.segm_size, self.link.segm_size)),
            _random_array(
                self.link.xp,
                (2, self.n_class, self.link.segm_size, self.link.segm_size)),
            _random_array(
                self.link.xp,
                (1, self.n_class, self.link.segm_size, self.link.segm_size))
        ]
        bboxes = [
            self.link.xp.array(((4, 1, 6, 3),), dtype=np.float32),
            self.link.xp.array(
                ((0, 1, 2, 3), (5, 4, 10, 6)), dtype=np.float32),
            self.link.xp.array(((10, 4, 12, 10),), dtype=np.float32),
        ]
        labels = [
            self.link.xp.random.randint(
                0, self.n_class - 1, size=(1,), dtype=np.int32),
            self.link.xp.random.randint(
                0, self.n_class - 1, size=(2,), dtype=np.int32),
            self.link.xp.random.randint(
                0, self.n_class - 1, size=(1,), dtype=np.int32),
        ]

        sizes = [(56, 56), (48, 48), (72, 72)]
        masks = self.link.decode(
            segms, bboxes, labels, sizes)

        self.assertEqual(len(masks), 3)
        for n in range(3):
            self.assertIsInstance(masks[n], self.link.xp.ndarray)

            self.assertEqual(masks[n].shape[0], labels[n].shape[0])
            self.assertEqual(masks[n].shape[1:], sizes[n])

    @unittest.skipUnless(_cv2_available, 'cv2 is not installed')
    def test_decode_cpu(self):
        self._check_decode()


class TestMaskHeadLoss(unittest.TestCase):

    def _check_mask_head_loss_pre(self, xp):
        n_inst = 12
        segm_size = 28
        rois = [
            xp.array(((4, 1, 6, 3),), dtype=np.float32),
            xp.array(
                ((0, 1, 2, 3), (5, 4, 10, 6)), dtype=np.float32),
            xp.array(((10, 4, 12, 10),), dtype=np.float32),
        ]
        roi_indices = [
            xp.array((0,), dtype=np.int32),
            xp.array((1, 0), dtype=np.int32),
            xp.array((1,), dtype=np.int32),
        ]
        masks = [
            _random_array(xp, (n_inst, 60, 70)),
            _random_array(xp, (n_inst, 60, 70)),
        ]
        bboxes = [mask_to_bbox(mask) for mask in masks]
        labels = [
            xp.array((1,), dtype=np.int32),
            xp.array((10, 4), dtype=np.int32),
            xp.array((3,), dtype=np.int32),
        ]
        rois, roi_indices, gt_segms, gt_mask_labels = mask_head_loss_pre(
            rois, roi_indices, masks, bboxes, labels, segm_size)

        self.assertEqual(len(rois), 3)
        self.assertEqual(len(roi_indices), 3)
        self.assertEqual(len(gt_segms), 3)
        self.assertEqual(len(gt_mask_labels), 3)
        for l in range(3):
            self.assertIsInstance(rois[l], xp.ndarray)
            self.assertIsInstance(roi_indices[l], xp.ndarray)
            self.assertIsInstance(gt_segms[l], xp.ndarray)
            self.assertIsInstance(gt_mask_labels[l], xp.ndarray)

            self.assertEqual(rois[l].shape[0], roi_indices[l].shape[0])
            self.assertEqual(rois[l].shape[0], gt_segms[l].shape[0])
            self.assertEqual(rois[l].shape[0], gt_mask_labels[l].shape[0])
            self.assertEqual(rois[l].shape[1:], (4,))
            self.assertEqual(roi_indices[l].shape[1:], ())
            self.assertEqual(gt_segms[l].shape[1:], (segm_size, segm_size))
            self.assertEqual(gt_mask_labels[l].shape[1:], ())
            self.assertEqual(gt_segms[l].dtype, np.float32)
            self.assertEqual(gt_mask_labels[l].dtype, np.int32)

    @unittest.skipUnless(_cv2_available, 'cv2 is not installed')
    def test_mask_head_loss_pre_cpu(self):
        self._check_mask_head_loss_pre(np)

    @attr.gpu
    @unittest.skipUnless(_cv2_available, 'cv2 is not installed')
    def test_mask_head_loss_pre_gpu(self):
        import cupy
        self._check_mask_head_loss_pre(cupy)

    def _check_head_loss_post(self, xp):
        B = 2
        segms = chainer.Variable(_random_array(xp, (20, 81, 28, 28)))
        mask_roi_indices = [
            xp.random.randint(0, B, size=5).astype(np.int32),
            xp.random.randint(0, B, size=7).astype(np.int32),
            xp.random.randint(0, B, size=8).astype(np.int32),
        ]
        gt_segms = [
            _random_array(xp, (5, 28, 28)),
            _random_array(xp, (7, 28, 28)),
            _random_array(xp, (8, 28, 28)),
        ]
        gt_mask_labels = [
            xp.random.randint(0, 80, size=5).astype(np.int32),
            xp.random.randint(0, 80, size=7).astype(np.int32),
            xp.random.randint(0, 80, size=8).astype(np.int32),
        ]

        mask_head_loss = mask_head_loss_post(
            segms, mask_roi_indices, gt_segms, gt_mask_labels, B)

        self.assertIsInstance(mask_head_loss, chainer.Variable)
        self.assertIsInstance(mask_head_loss.array, xp.ndarray)
        self.assertEqual(mask_head_loss.shape, ())

    def test_head_loss_post_cpu(self):
        self._check_head_loss_post(np)

    @attr.gpu
    def test_head_loss_post_gpu(self):
        import cupy
        self._check_head_loss_post(cupy)


testing.run_module(__name__, __file__)
