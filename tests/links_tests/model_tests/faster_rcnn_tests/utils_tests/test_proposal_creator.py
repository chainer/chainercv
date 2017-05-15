import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links import ProposalCreator


def _generate_bbox(n, img_size, min_length, max_length):
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


@testing.parameterize(
    *testing.product({
        'train': [True, False],
        'use_gpu_nms': [True]
    })
)
class TestProposalCreator(unittest.TestCase):

    img_size = (320, 240)
    rpn_batch_size = 256
    n_anchor_base = 9
    train_rpn_post_nms_top_n = 350
    test_rpn_post_nms_top_n = 300

    def setUp(self):
        n_anchor_base = 9
        feat_size = (self.img_size[0] // 16, self.img_size[1] // 16)
        n_anchor = np.int32(self.n_anchor_base * np.prod(feat_size))

        self.rpn_cls_prob = np.random.uniform(
            low=0., high=1.,
            size=(1, 2 * n_anchor_base) + feat_size).astype(np.float32)

        self.rpn_bbox_pred = np.random.uniform(
            low=0., high=1.,
            size=(1, 4 * n_anchor_base) + feat_size).astype(np.float32)
        self.anchor = _generate_bbox(n_anchor, self.img_size, 16, 200)
        self.proposal_creator = ProposalCreator(
            use_gpu_nms=self.use_gpu_nms,
            train_rpn_post_nms_top_n=self.train_rpn_post_nms_top_n,
            test_rpn_post_nms_top_n=self.test_rpn_post_nms_top_n,
            rpn_min_size=0)

    def check_proposal_creator(
            self, proposal_creator,
            rpn_bbox_pred, rpn_cls_prob, anchor, img_size,
            scale=1., train=False):
        xp = cuda.get_array_module(rpn_bbox_pred)
        rois, batch_indices = self.proposal_creator(
            rpn_bbox_pred, rpn_cls_prob, anchor, img_size, scale, train)

        out_length = self.train_rpn_post_nms_top_n \
            if train else self.test_rpn_post_nms_top_n
        self.assertIsInstance(rois, xp.ndarray)
        self.assertEqual(rois.shape, (out_length, 4))
        self.assertIsInstance(batch_indices, xp.ndarray)
        self.assertEqual(batch_indices.shape, (out_length,))
        np.testing.assert_equal(
            cuda.to_cpu(batch_indices),
            np.zeros((len(batch_indices),), dtype=np.int32))

    def test_proposal_creator_cpu(self):
        self.check_proposal_creator(
            self.proposal_creator,
            chainer.Variable(self.rpn_bbox_pred),
            chainer.Variable(self.rpn_cls_prob),
            self.anchor, self.img_size, scale=1., train=self.train)

    @attr.gpu
    def test_proposal_creator_gpu(self):
        self.check_proposal_creator(
            self.proposal_creator,
            chainer.Variable(cuda.to_gpu(self.rpn_bbox_pred)),
            chainer.Variable(cuda.to_gpu(self.rpn_cls_prob)),
            cuda.to_gpu(self.anchor), self.img_size,
            scale=1., train=self.train)


testing.run_module(__name__, __file__)
