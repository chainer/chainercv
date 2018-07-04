import unittest

import numpy as np

from chainer.backends import cuda

from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links.model.fcis import ProposalTargetCreator
from chainercv.utils import generate_random_bbox
from chainercv.utils import mask_to_bbox


class TestProposalTargetCreator(unittest.TestCase):

    n_sample = 128
    n_class = 21
    pos_ratio = 0.25
    mask_size = 21

    def setUp(self):

        n_roi = 1024
        n_mask = 10
        img_size = (392, 512)
        self.roi = generate_random_bbox(n_roi, img_size, 16, 250)
        self.mask = np.random.uniform(
            size=(n_mask, img_size[0], img_size[1])) > 0.5
        self.label = np.random.randint(
            0, self.n_class - 1, size=(n_mask,), dtype=np.int32)

        self.proposal_target_creator = ProposalTargetCreator(
            n_sample=self.n_sample,
            pos_ratio=self.pos_ratio)

    def check_proposal_target_creator(
            self, roi, mask, label, proposal_target_creator):
        xp = cuda.get_array_module(roi)
        bbox = mask_to_bbox(mask)
        sample_roi, gt_roi_mask, gt_roi_label, gt_roi_loc =\
            proposal_target_creator(
                roi, mask, label, bbox, mask_size=self.mask_size)

        # Test types
        self.assertIsInstance(sample_roi, xp.ndarray)
        self.assertIsInstance(gt_roi_loc, xp.ndarray)
        self.assertIsInstance(gt_roi_mask, xp.ndarray)
        self.assertIsInstance(gt_roi_label, xp.ndarray)

        sample_roi = cuda.to_cpu(sample_roi)
        gt_roi_loc = cuda.to_cpu(gt_roi_loc)
        gt_roi_mask = cuda.to_cpu(gt_roi_mask)
        gt_roi_label = cuda.to_cpu(gt_roi_label)

        # Test shapes
        self.assertEqual(sample_roi.shape, (self.n_sample, 4))
        self.assertEqual(gt_roi_loc.shape, (self.n_sample, 4))
        self.assertEqual(
            gt_roi_mask.shape, (self.n_sample, self.mask_size, self.mask_size))
        self.assertEqual(gt_roi_label.shape, (self.n_sample,))

        # Test foreground and background labels
        np.testing.assert_equal(np.sum(gt_roi_label >= 0), self.n_sample)
        n_pos = np.sum(gt_roi_label >= 1)
        n_neg = np.sum(gt_roi_label == 0)
        self.assertLessEqual(n_pos, self.n_sample * self.pos_ratio)
        self.assertLessEqual(n_neg, self.n_sample - n_pos)

    def test_proposal_target_creator_cpu(self):
        self.check_proposal_target_creator(
            self.roi, self.mask, self.label,
            self.proposal_target_creator)

    @attr.gpu
    def test_proposal_target_creator_gpu(self):
        self.check_proposal_target_creator(
            cuda.to_gpu(self.roi),
            cuda.to_gpu(self.mask),
            cuda.to_gpu(self.label),
            self.proposal_target_creator)


testing.run_module(__name__, __file__)
