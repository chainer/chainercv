import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain


def _generate_bbox(xp, n, img_size, min_length, max_length):
    W, H = img_size
    x_min = xp.random.uniform(0, W - max_length, size=(n,))
    y_min = xp.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + xp.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + xp.random.uniform(min_length, max_length, size=(n,))
    bbox = xp.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


@testing.parameterize(
    {'train': False},
    {'train': True}
)
@attr.slow
class TestFasterRCNNVGG16(unittest.TestCase):

    B = 2
    n_fg_class = 20
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
            self.n_fg_class, pretrained_model=None,
            proposal_creator_params=proposal_creator_params)

    def check_call(self):
        xp = self.link.xp

        feat_size = (12, 16)
        x = chainer.Variable(
            xp.random.uniform(
                low=-1., high=1.,
                size=(self.B, 3, feat_size[1] * 16, feat_size[0] * 16)
            ).astype(np.float32), volatile=chainer.flag.ON)
        roi_cls_locs, roi_scores, rois, roi_indices = self.link(
            x, test=not self.train)
        if self.train:
            n_roi = self.B * self.n_train_post_nms
        else:
            n_roi = self.B * self.n_test_post_nms

        self.assertIsInstance(roi_cls_locs, chainer.Variable)
        self.assertIsInstance(roi_cls_locs.data, xp.ndarray)
        self.assertEqual(roi_cls_locs.shape, (n_roi, self.n_class * 4))

        self.assertIsInstance(roi_scores, chainer.Variable)
        self.assertIsInstance(roi_scores.data, xp.ndarray)
        self.assertEqual(roi_scores.shape, (n_roi, self.n_class))

        self.assertIsInstance(rois, xp.ndarray)
        self.assertEqual(rois.shape, (n_roi, 4))

        self.assertIsInstance(roi_indices, xp.ndarray)
        self.assertEqual(roi_indices.shape, (n_roi,))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


@attr.slow
class TestFasterRCNNVGG16Loss(unittest.TestCase):

    n_fg_class = 20

    def setUp(self):
        faster_rcnn = FasterRCNNVGG16(
            n_fg_class=self.n_fg_class, pretrained_model=False)
        self.link = FasterRCNNTrainChain(faster_rcnn)

    def check_call(self):
        xp = self.link.xp

        n_bbox = 3
        imgs = xp.random.uniform(
            low=-122.5, high=122.5, size=(1, 3, 600, 800)).astype(np.float32)
        imgs = chainer.Variable(imgs)
        bboxes = chainer.Variable(
            _generate_bbox(xp, n_bbox, (600, 800), 16, 350)[None])
        labels = xp.random.randint(0, self.n_fg_class + 1, size=(n_bbox,))
        labels = chainer.Variable(labels[None].astype(np.int32))
        scale = chainer.Variable(xp.array(1.))
        loss = self.link(imgs, bboxes, labels, scale)
        self.assertEqual(loss.shape, ())

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
