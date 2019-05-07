import numpy as np
import unittest

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links import FCISResNet101
from chainercv.experimental.links.model.fcis import FCISTrainChain
from chainercv.utils import mask_to_bbox

from tests.experimental_tests.links_tests.model_tests.fcis_tests.test_fcis \
    import _random_array


@testing.parameterize(
    {'train': False, 'iter2': True},
    {'train': True, 'iter2': False}
)
class TestFCISResNet101(unittest.TestCase):

    B = 1
    n_fg_class = 20
    n_class = n_fg_class + 1
    n_anchor = 9
    n_train_post_nms = 12
    n_test_post_nms = 8

    def setUp(self):
        proposal_creator_params = {
            'n_train_post_nms': self.n_train_post_nms,
            'n_test_post_nms': self.n_test_post_nms,
        }
        self.link = FCISResNet101(
            self.n_fg_class, pretrained_model=None,
            iter2=self.iter2,
            proposal_creator_params=proposal_creator_params)

    def check_call(self):
        xp = self.link.xp

        feat_size = (12, 16)
        x = chainer.Variable(
            xp.random.uniform(
                low=-1., high=1.,
                size=(self.B, 3, feat_size[0] * 16, feat_size[1] * 16)
            ).astype(np.float32))
        with chainer.using_config('train', self.train):
            (roi_ag_seg_scores, roi_ag_locs, roi_cls_scores,
             rois, roi_indices) = self.link(x)

        n_roi = roi_ag_seg_scores.shape[0]
        if self.train:
            self.assertGreaterEqual(self.B * self.n_train_post_nms, n_roi)
        else:
            self.assertGreaterEqual(self.B * self.n_test_post_nms * 2, n_roi)

        self.assertIsInstance(roi_ag_seg_scores, chainer.Variable)
        self.assertIsInstance(roi_ag_seg_scores.array, xp.ndarray)
        self.assertEqual(
            roi_ag_seg_scores.shape, (n_roi, 2, 21, 21))

        self.assertIsInstance(roi_ag_locs, chainer.Variable)
        self.assertIsInstance(roi_ag_locs.array, xp.ndarray)
        self.assertEqual(roi_ag_locs.shape, (n_roi, 2, 4))

        self.assertIsInstance(roi_cls_scores, chainer.Variable)
        self.assertIsInstance(roi_cls_scores.array, xp.ndarray)
        self.assertEqual(roi_cls_scores.shape, (n_roi, self.n_class))

        self.assertIsInstance(rois, xp.ndarray)
        self.assertEqual(rois.shape, (n_roi, 4))

        self.assertIsInstance(roi_indices, xp.ndarray)
        self.assertEqual(roi_indices.shape, (n_roi,))

    @attr.slow
    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


class TestFCISResNet101Loss(unittest.TestCase):

    B = 1
    n_fg_class = 20
    n_bbox = 3
    n_anchor = 9
    n_train_post_nms = 12
    n_test_post_nms = 8

    def setUp(self):
        proposal_creator_params = {
            'n_train_post_nms': self.n_train_post_nms,
            'n_test_post_nms': self.n_test_post_nms,
        }
        self.model = FCISTrainChain(
            FCISResNet101(
                self.n_fg_class, pretrained_model=None, iter2=False,
                proposal_creator_params=proposal_creator_params))

        self.masks = np.random.randint(
            0, 2, size=(1, self.n_bbox, 600, 800)).astype(np.bool)
        self.labels = np.random.randint(
            0, self.n_fg_class, size=(1, self.n_bbox)).astype(np.int32)
        self.imgs = _random_array(np, (1, 3, 600, 800))
        self.scale = np.array(1.)

    def check_call(self, model, imgs, masks, labels, scale):
        bboxes = mask_to_bbox(masks[0])[None]
        loss = model(imgs, masks, labels, bboxes, scale)
        self.assertEqual(loss.shape, ())

    def test_call_cpu(self):
        self.check_call(
            self.model, self.imgs, self.masks, self.labels, self.scale)

    @attr.gpu
    def test_call_gpu(self):
        self.model.to_gpu()
        self.check_call(
            self.model, cuda.to_gpu(self.imgs),
            self.masks, self.labels, self.scale)


@testing.parameterize(*testing.product({
    'n_fg_class': [None, 10, 20, 80],
    'anchor_scales': [(8, 16, 32), (4, 8, 16, 32)],
    'pretrained_model': ['sbd', 'sbd_converted', 'coco', 'coco_converted'],
}))
class TestFCISResNet101Pretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_fg_class': self.n_fg_class,
            'anchor_scales': self.anchor_scales,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model.startswith('sbd'):
            valid = self.n_fg_class in [None, 20]
            valid = valid and self.anchor_scales == (8, 16, 32)
        elif self.pretrained_model.startswith('coco'):
            valid = self.n_fg_class in [None, 80]
            valid = valid and self.anchor_scales == (4, 8, 16, 32)

        if valid:
            FCISResNet101(**kwargs)
        else:
            with self.assertRaises(ValueError):
                FCISResNet101(**kwargs)


testing.run_module(__name__, __file__)
