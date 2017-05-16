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

    B = 2
    n_class = 21
    n_anchor = 9
    n_train_post_nms = 12
    n_test_post_nms = 8
    n_conv5_3_channel = 512

    def setUp(self):
        proposal_creator_params = {
            'n_train_post_nms': self.n_train_post_nms,
            'n_test_post_nms': self.n_test_post_nms
        }
        self.link = FasterRCNNVGG16(
            self.n_class, proposal_creator_params=proposal_creator_params)

    def check_call(self):
        xp = self.link.xp

        feat_size = (12, 16)
        x = chainer.Variable(
            xp.random.uniform(
                low=-1., high=1.,
                size=(self.B, 3, feat_size[1] * 16, feat_size[0] * 16)
            ).astype(np.float32), volatile=chainer.flag.ON)
        y = self.link(
            x,
            layers=['features', 'rpn_bboxes', 'rpn_scores',
                    'rois', 'batch_indices', 'anchor',
                    'roi_bboxes', 'roi_scores'],
            test=not self.train
        )
        if self.train:
            n_roi = self.B * self.n_train_post_nms
        else:
            n_roi = self.B * self.n_test_post_nms

        self.assertIsInstance(y['features'], chainer.Variable)
        self.assertIsInstance(y['features'].data, xp.ndarray)
        self.assertEqual(
            y['features'].shape,
            (self.B, self.n_conv5_3_channel, feat_size[1], feat_size[0]))

        self.assertIsInstance(y['rpn_bboxes'], chainer.Variable)
        self.assertIsInstance(y['rpn_bboxes'].data, xp.ndarray)
        self.assertEqual(
            y['rpn_bboxes'].shape,
            (self.B, self.n_anchor * 4, feat_size[1], feat_size[0]))

        self.assertIsInstance(y['rpn_scores'], chainer.Variable)
        self.assertIsInstance(y['rpn_scores'].data, xp.ndarray)
        self.assertEqual(
            y['rpn_scores'].shape,
            (self.B, self.n_anchor * 2, feat_size[1], feat_size[0]))

        self.assertIsInstance(y['rois'], xp.ndarray)
        self.assertEqual(y['rois'].shape, (n_roi, 4))

        self.assertIsInstance(y['anchor'], xp.ndarray)
        self.assertEqual(
            y['anchor'].shape,
            (self.n_anchor * feat_size[1] * feat_size[0], 4))

        self.assertIsInstance(y['roi_bboxes'], chainer.Variable)
        self.assertIsInstance(y['roi_bboxes'].data, xp.ndarray)
        self.assertEqual(
            y['roi_bboxes'].shape, (n_roi, self.n_class * 4))

        self.assertIsInstance(y['roi_scores'], chainer.Variable)
        self.assertIsInstance(y['roi_scores'].data, xp.ndarray)
        self.assertEqual(
            y['roi_scores'].shape, (n_roi, self.n_class))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
