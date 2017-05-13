import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import FasterRCNNVGG16


@testing.parameterize(
    {'train': False},
    {'train': True}
)
@attr.slow
class TestFasterRCNNVGG16(unittest.TestCase):

    n_class = 21
    n_anchor = 9
    n_train_roi = 12
    n_test_roi = 8
    n_conv5_3_channel = 512

    def setUp(self):
        proposal_creator_params = {
            'train_rpn_post_nms_top_n': self.n_train_roi,
            'test_rpn_post_nms_top_n': self.n_test_roi
        }
        self.link = FasterRCNNVGG16(
            self.n_class, proposal_creator_params=proposal_creator_params)

    def check_call(self):
        xp = self.link.xp

        feat_size = (12, 16)
        x = chainer.Variable(
            xp.random.uniform(
                low=-1., high=1.,
                size=(1, 3, feat_size[1] * 16, feat_size[0] * 16)
            ).astype(np.float32))
        y = self.link(
            x,
            layers=['feature', 'rpn_bbox_pred', 'rpn_cls_score',
                    'roi', 'anchor', 'pool', 'bbox_tf', 'score'],
            test=not self.train
        )
        if self.train:
            n_roi = self.n_train_roi
        else:
            n_roi = self.n_test_roi

        self.assertIsInstance(y['feature'], chainer.Variable)
        self.assertIsInstance(y['feature'].data, xp.ndarray)
        self.assertEqual(
            y['feature'].shape,
            (1, self.n_conv5_3_channel, feat_size[1], feat_size[0]))

        self.assertIsInstance(y['rpn_bbox_pred'], chainer.Variable)
        self.assertIsInstance(y['rpn_bbox_pred'].data, xp.ndarray)
        self.assertEqual(
            y['rpn_bbox_pred'].shape,
            (1, self.n_anchor * 4, feat_size[1], feat_size[0]))

        self.assertIsInstance(y['rpn_cls_score'], chainer.Variable)
        self.assertIsInstance(y['rpn_cls_score'].data, xp.ndarray)
        self.assertEqual(
            y['rpn_cls_score'].shape,
            (1, self.n_anchor * 2, feat_size[1], feat_size[0]))

        self.assertIsInstance(y['roi'], xp.ndarray)
        self.assertEqual(y['roi'].shape, (n_roi, 5))

        self.assertIsInstance(y['anchor'], xp.ndarray)
        self.assertEqual(
            y['anchor'].shape,
            (self.n_anchor * feat_size[1] * feat_size[0], 4))

        self.assertIsInstance(y['pool'], chainer.Variable)
        self.assertIsInstance(y['pool'].data, xp.ndarray)
        self.assertEqual(
            y['pool'].shape, (n_roi, self.n_conv5_3_channel, 7, 7))

        self.assertIsInstance(y['bbox_tf'], chainer.Variable)
        self.assertIsInstance(y['bbox_tf'].data, xp.ndarray)
        self.assertEqual(
            y['bbox_tf'].shape, (n_roi, self.n_class * 4))

        self.assertIsInstance(y['score'], chainer.Variable)
        self.assertIsInstance(y['score'].data, xp.ndarray)
        self.assertEqual(
            y['score'].shape, (n_roi, self.n_class))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
