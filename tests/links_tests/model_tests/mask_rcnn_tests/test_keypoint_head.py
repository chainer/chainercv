from __future__ import division

import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.mask_rcnn import KeypointHead
from chainercv.links.model.mask_rcnn import keypoint_loss_post
from chainercv.links.model.mask_rcnn import keypoint_loss_pre


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


def _point_to_bbox(point, visible=None):
    xp = chainer.backends.cuda.get_array_module(point)

    bbox = xp.zeros((len(point), 4), dtype=np.float32)

    for i, pnt in enumerate(point):
        if visible is None:
            vsbl = xp.ones((len(pnt),), dtype=np.bool)
        else:
            vsbl = visible[i]
        pnt = pnt[vsbl]
        bbox[i, 0] = xp.min(pnt[:, 0])
        bbox[i, 1] = xp.min(pnt[:, 1])
        bbox[i, 2] = xp.max(pnt[:, 0])
        bbox[i, 3] = xp.max(pnt[:, 1])
    return bbox


class TestKeypointHeadLoss(unittest.TestCase):

    def _check_keypoint_loss_pre(self, xp):
        point_map_size = 28
        n_point = 17
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
        points = [
            xp.zeros((1, n_point, 2), dtype=np.float32),
            xp.zeros((3, n_point, 2), dtype=np.float32),
        ]
        visibles = [
            xp.ones((1, n_point), dtype=np.bool),
            xp.ones((3, n_point), dtype=np.bool),
        ]
        bboxes = [_point_to_bbox(point, visible)
                  for point, visible in zip(points, visibles)]
        labels = [
            xp.array((1,), dtype=np.int32),
            xp.array((1, 1), dtype=np.int32),
            xp.array((1,), dtype=np.int32),
        ]
        rois, roi_indices, gt_roi_points, gt_roi_visibles = keypoint_loss_pre(
            rois, roi_indices, points, visibles, bboxes,
            labels, point_map_size)

        self.assertEqual(len(rois), 3)
        self.assertEqual(len(roi_indices), 3)
        self.assertEqual(len(gt_roi_points), 3)
        self.assertEqual(len(gt_roi_visibles), 3)
        for l in range(3):
            self.assertIsInstance(rois[l], xp.ndarray)
            self.assertIsInstance(roi_indices[l], xp.ndarray)
            self.assertIsInstance(gt_roi_points[l], xp.ndarray)
            self.assertIsInstance(gt_roi_visibles[l], xp.ndarray)

            self.assertEqual(rois[l].shape[0], roi_indices[l].shape[0])
            self.assertEqual(rois[l].shape[0], gt_roi_points[l].shape[0])
            self.assertEqual(rois[l].shape[0], gt_roi_visibles[l].shape[0])
            self.assertEqual(rois[l].shape[1:], (4,))
            self.assertEqual(roi_indices[l].shape[1:], ())
            self.assertEqual(
                gt_roi_points[l].shape[1:], (n_point, 2))
            self.assertEqual(
                gt_roi_visibles[l].shape[1:], (n_point,))

            self.assertEqual(
                gt_roi_points[l].dtype, np.float32)
            self.assertEqual(
                gt_roi_visibles[l].dtype, np.bool)

    def test_keypoint_loss_pre_cpu(self):
        self._check_keypoint_loss_pre(np)

    @attr.gpu
    def test_keypoint_loss_pre_gpu(self):
        import cupy
        self._check_keypoint_loss_pre(cupy)

    def _check_keypoint_loss_post(self, xp):
        B = 2
        n_point = 17

        point_maps = chainer.Variable(_random_array(xp, (20, n_point, 28, 28)))
        point_roi_indices = [
            xp.random.randint(0, B, size=5).astype(np.int32),
            xp.random.randint(0, B, size=7).astype(np.int32),
            xp.random.randint(0, B, size=8).astype(np.int32),
        ]
        gt_roi_points = [
            xp.random.randint(0, 28, size=(5, n_point, 2)).astype(np.int32),
            xp.random.randint(0, 28, size=(7, n_point, 2)).astype(np.int32),
            xp.random.randint(0, 28, size=(8, n_point, 2)).astype(np.int32),
        ]
        gt_roi_visibles = [
            xp.random.randint(0, 2, size=(5, n_point)).astype(np.bool),
            xp.random.randint(0, 2, size=(7, n_point)).astype(np.bool),
            xp.random.randint(0, 2, size=(8, n_point)).astype(np.bool),
        ]

        keypoint_loss = keypoint_loss_post(
            point_maps, point_roi_indices, gt_roi_points,
            gt_roi_visibles, B)

        self.assertIsInstance(keypoint_loss, chainer.Variable)
        self.assertIsInstance(keypoint_loss.array, xp.ndarray)
        self.assertEqual(keypoint_loss.shape, ())

    def test_keypoint_loss_post_cpu(self):
        self._check_keypoint_loss_post(np)

    @attr.gpu
    def test_keypoint_loss_post_gpu(self):
        import cupy
        self._check_keypoint_loss_post(cupy)


testing.run_module(__name__, __file__)
