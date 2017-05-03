import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainercv.links.faster_rcnn.proposal_layer import ProposalLayer


def generate_bbox(n, img_size, min_length, max_length):
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
        'use_gpu_nms': [True, False]
    })
)
class TestProposalLayer(unittest.TestCase):

    def setUp(self):
        n_anchor_base = 9
        img_size = (320, 240)
        feat_size = (img_size[0] / 16, img_size[1] / 16)
        n_anchor = n_anchor_base * np.prod(feat_size)
        self.train_rpn_post_nms_top_n = 350
        self.test_rpn_post_nms_top_n = 300

        self.rpn_cls_prob = np.random.uniform(
            low=0., high=1.,
            size=(1, 2 * n_anchor_base) + feat_size).astype(np.float32)

        self.rpn_bbox_pred = np.random.uniform(
            low=0., high=1.,
            size=(1, 4 * n_anchor_base) + feat_size).astype(np.float32)
        self.anchor = generate_bbox(n_anchor, img_size, 2, 5)
        self.img_size = img_size
        self.proposal_layer = ProposalLayer(
            use_gpu_nms=self.use_gpu_nms,
            train_rpn_post_nms_top_n=self.train_rpn_post_nms_top_n,
            test_rpn_post_nms_top_n=self.test_rpn_post_nms_top_n,
            rpn_min_size=0)

    def check_proposal_layer(
            self, proposal_layer,
            rpn_bbox_pred, rpn_cls_prob, anchor, img_size,
            scale=1., train=False):
        xp = cuda.get_array_module(rpn_bbox_pred)
        roi = self.proposal_layer(
            rpn_bbox_pred, rpn_cls_prob, anchor, img_size, scale, train)

        out_length = self.train_rpn_post_nms_top_n \
            if train else self.test_rpn_post_nms_top_n
        self.assertEqual(roi.shape[0], out_length)
        self.assertIsInstance(roi, xp.ndarray)

    @condition.retry(3)
    def test_proposal_layer_cpu(self):
        self.check_proposal_layer(
            self.proposal_layer,
            chainer.Variable(self.rpn_bbox_pred),
            chainer.Variable(self.rpn_cls_prob),
            self.anchor, self.img_size, scale=1., train=self.train)

    @attr.gpu
    @condition.retry(3)
    def test_proposal_layer_gpu(self):
        self.check_proposal_layer(
            self.proposal_layer,
            chainer.Variable(cuda.to_gpu(self.rpn_bbox_pred)),
            chainer.Variable(cuda.to_gpu(self.rpn_cls_prob)),
            cuda.to_gpu(self.anchor), self.img_size,
            scale=1., train=self.train)


testing.run_module(__name__, __file__)
