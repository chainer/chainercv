import unittest

import numpy as np

from chainer import cuda

from chainer import testing
from chainer.testing import attr

from chainercv.links.model.faster_rcnn import ProposalTargetCreator


def _generate_bbox(n, img_size, min_length, max_length):
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


class TestProposalTargetCreator(unittest.TestCase):

    batch_size = 128
    n_class = 21
    fg_fraction = 0.25
    loc_in_weight = (0.7, 0.8, 0.9, 1.)

    def setUp(self):

        n_roi = 1024
        n_bbox = 10
        self.roi = _generate_bbox(n_roi, (392, 512), 16, 250)
        self.bbox = _generate_bbox(n_bbox, (392, 512), 16, 250)
        self.label = np.random.randint(
            0, self.n_class, size=(n_bbox,), dtype=np.int32)

        self.proposal_target_creator = ProposalTargetCreator(
            batch_size=self.batch_size,
            fg_fraction=self.fg_fraction,
            loc_in_weight=self.loc_in_weight,
        )

    def check_proposal_target_creator(
            self, bbox, label, roi, proposal_target_creator):
        xp = cuda.get_array_module(roi)
        (sample_roi, roi_bbox_target, roi_gt_label, roi_loc_in_weight,
         roi_bbox_outside_weight) =\
            proposal_target_creator(roi, bbox, label, self.n_class)

        # Test types
        self.assertIsInstance(sample_roi, xp.ndarray)
        self.assertIsInstance(roi_bbox_target, xp.ndarray)
        self.assertIsInstance(roi_gt_label, xp.ndarray)
        self.assertIsInstance(roi_loc_in_weight, xp.ndarray)
        self.assertIsInstance(roi_bbox_outside_weight, xp.ndarray)

        sample_roi = cuda.to_cpu(sample_roi)
        roi_bbox_target = cuda.to_cpu(roi_bbox_target)
        roi_gt_label = cuda.to_cpu(roi_gt_label)
        roi_loc_in_weight = cuda.to_cpu(roi_loc_in_weight)
        roi_bbox_outside_weight = cuda.to_cpu(roi_bbox_outside_weight)

        # Test shapes
        self.assertEqual(sample_roi.shape,
                         (self.batch_size, 4))
        self.assertEqual(roi_bbox_target.shape,
                         (self.batch_size, 4 * self.n_class))
        self.assertEqual(roi_gt_label.shape,
                         (self.batch_size,))
        self.assertEqual(roi_loc_in_weight.shape,
                         (self.batch_size, 4 * self.n_class))
        self.assertEqual(roi_bbox_outside_weight.shape,
                         (self.batch_size, 4 * self.n_class))

        # Test foreground and background labels
        np.testing.assert_equal(
            np.sum(roi_gt_label >= 0),
            self.batch_size)
        n_fg = np.sum(roi_gt_label >= 1)
        n_bg = np.sum(roi_gt_label == 0)
        self.assertLessEqual(
            n_fg, self.batch_size * self.fg_fraction)
        self.assertLessEqual(n_bg, self.batch_size - n_fg)

        # Test roi_loc_in_weight and bbox_outside_weight
        box_index_0 = np.where(roi_gt_label >= 1)[0][0]
        index_0 = (
            box_index_0,
            slice(4 * roi_gt_label[box_index_0],
                  4 * roi_gt_label[box_index_0] + 4))
        bbox_inside_00 = roi_loc_in_weight[index_0]
        bbox_outside_00 = roi_bbox_outside_weight[index_0]
        np.testing.assert_equal(
            bbox_inside_00,
            np.array(self.loc_in_weight, dtype=np.float32))
        np.testing.assert_equal(
            bbox_outside_00,
            np.array((1, 1, 1, 1), dtype=np.float32))

    def test_proposal_target_creator_cpu(self):
        self.check_proposal_target_creator(
            self.bbox, self.label, self.roi,
            self.proposal_target_creator)

    @attr.gpu
    def test_proposal_target_creator_gpu(self):
        self.check_proposal_target_creator(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.label),
            cuda.to_gpu(self.roi),
            self.proposal_target_creator)


testing.run_module(__name__, __file__)
