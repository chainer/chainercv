import unittest

import numpy as np

from chainer import cuda

from chainer import testing
from chainer.testing import attr

from chainercv.links import ProposalTargetCreator


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
    bbox_inside_weight = (0.7, 0.8, 0.9, 1.)

    def setUp(self):

        n_roi = 1024
        n_bbox = 10
        roi = _generate_bbox(n_roi, (392, 512), 16, 250)
        self.roi = np.concatenate((np.zeros((n_roi, 1)), roi), axis=1)
        self.bbox = _generate_bbox(n_bbox, (392, 512), 16, 250)[None]
        self.label = np.random.randint(
            0, self.n_class, size=(1, n_bbox,), dtype=np.int32)

        self.proposal_target_creator = ProposalTargetCreator(
            n_class=self.n_class,
            batch_size=self.batch_size,
            fg_fraction=self.fg_fraction,
            bbox_inside_weight=self.bbox_inside_weight,
        )

    def check_proposal_target_creator(
            self, roi, bbox, label, proposal_target_creator):
        xp = cuda.get_array_module(roi)
        (roi_sample, bbox_target_sample, label_sample, bbox_inside_weight,
         bbox_outside_weight) =\
            proposal_target_creator(roi, bbox, label)

        # Test types
        self.assertIsInstance(roi_sample, xp.ndarray)
        self.assertIsInstance(bbox_target_sample, xp.ndarray)
        self.assertIsInstance(label_sample, xp.ndarray)
        self.assertIsInstance(bbox_inside_weight, xp.ndarray)
        self.assertIsInstance(bbox_outside_weight, xp.ndarray)

        roi_sample = cuda.to_cpu(roi_sample)
        bbox_target_sample = cuda.to_cpu(bbox_target_sample)
        label_sample = cuda.to_cpu(label_sample)
        bbox_inside_weight = cuda.to_cpu(bbox_inside_weight)
        bbox_outside_weight = cuda.to_cpu(bbox_outside_weight)

        # Test shapes
        self.assertEqual(roi_sample.shape,
                         (self.batch_size, 5))
        self.assertEqual(bbox_target_sample.shape,
                         (1, self.batch_size, 4 * self.n_class))
        self.assertEqual(label_sample.shape,
                         (1, self.batch_size))
        self.assertEqual(bbox_inside_weight.shape,
                         (1, self.batch_size, 4 * self.n_class))
        self.assertEqual(bbox_outside_weight.shape,
                         (1, self.batch_size, 4 * self.n_class))

        # Test foreground and background labels
        np.testing.assert_equal(
            np.sum(label_sample >= 0),
            self.batch_size)
        n_fg = np.sum(label_sample >= 1)
        n_bg = np.sum(label_sample == 0)
        self.assertLessEqual(
            n_fg, self.batch_size * self.fg_fraction)
        self.assertLessEqual(n_bg, self.batch_size - n_fg)
        self.assertTrue(np.all(label_sample[n_fg:] == 0))

        # Test bbox_inside_weight and bbox_outside_weight
        box_index = np.where(label_sample[0] >= 1)[0][0]
        index = (
            0, box_index,
            slice(4 * label_sample[0, box_index],
                  4 * label_sample[0, box_index] + 4))
        bbox_inside_00 = bbox_inside_weight[index]
        bbox_outside_00 = bbox_outside_weight[index]
        np.testing.assert_equal(
            bbox_inside_00,
            np.array(self.bbox_inside_weight, dtype=np.float32))
        np.testing.assert_equal(
            bbox_outside_00,
            np.array((1, 1, 1, 1), dtype=np.float32))

    def test_proposal_target_creator_cpu(self):
        self.check_proposal_target_creator(
            self.roi, self.bbox, self.label, self.proposal_target_creator)

    @attr.gpu
    def test_proposal_target_creator_gpu(self):
        self.check_proposal_target_creator(
            cuda.to_gpu(self.roi),
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.label),
            self.proposal_target_creator)


testing.run_module(__name__, __file__)
