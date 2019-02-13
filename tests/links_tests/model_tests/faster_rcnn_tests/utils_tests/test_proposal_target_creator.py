import unittest

import numpy as np

from chainer.backends import cuda

from chainer import testing
from chainer.testing import attr

from chainercv.links.model.faster_rcnn import ProposalTargetCreator
from chainercv.utils import generate_random_bbox


@testing.parameterize(
    {'n_sample': 128}, {'n_sample': None}
)
class TestProposalTargetCreator(unittest.TestCase):

    n_class = 21
    pos_ratio = 0.25
    n_roi = 1024
    n_bbox = 10

    def setUp(self):

        self.roi = generate_random_bbox(self.n_roi, (392, 512), 16, 250)
        self.bbox = generate_random_bbox(self.n_bbox, (392, 512), 16, 250)
        self.label = np.random.randint(
            0, self.n_class - 1, size=(self.n_bbox,), dtype=np.int32)

        self.proposal_target_creator = ProposalTargetCreator(
            n_sample=self.n_sample,
            pos_ratio=self.pos_ratio,
        )

    def check_proposal_target_creator(
            self, bbox, label, roi, proposal_target_creator):
        xp = cuda.get_array_module(roi)
        sample_roi, gt_roi_loc, gt_roi_label =\
            proposal_target_creator(roi, bbox, label)

        # Test types
        self.assertIsInstance(sample_roi, xp.ndarray)
        self.assertIsInstance(gt_roi_loc, xp.ndarray)
        self.assertIsInstance(gt_roi_label, xp.ndarray)

        sample_roi = cuda.to_cpu(sample_roi)
        gt_roi_loc = cuda.to_cpu(gt_roi_loc)
        gt_roi_label = cuda.to_cpu(gt_roi_label)

        # Test shapes
        if self.n_sample is None:
            n_sample = self.n_roi + self.n_bbox
        else:
            n_sample = self.n_sample
        self.assertEqual(sample_roi.shape, (n_sample, 4))
        self.assertEqual(gt_roi_loc.shape, (n_sample, 4))
        self.assertEqual(gt_roi_label.shape, (n_sample,))

        # Test foreground and background labels
        np.testing.assert_equal(np.sum(gt_roi_label >= 0), n_sample)
        n_pos = np.sum(gt_roi_label >= 1)
        n_neg = np.sum(gt_roi_label == 0)
        self.assertLessEqual(n_pos, n_sample * self.pos_ratio)
        self.assertLessEqual(n_neg, n_sample - n_pos)

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
