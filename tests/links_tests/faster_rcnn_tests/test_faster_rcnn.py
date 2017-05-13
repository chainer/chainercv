import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import FasterRCNNBase


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


def _generate_bbox(xp, n, img_size, min_length, max_length):
    W, H = img_size
    x_min = xp.random.uniform(0, W - max_length, size=(n,))
    y_min = xp.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + xp.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + xp.random.uniform(min_length, max_length, size=(n,))
    bbox = xp.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


class DummyFeature(chainer.Link):

    def __init__(self, feat_stride):
        super(DummyFeature, self).__init__()
        self.feat_stride = feat_stride

    def __call__(self, x, train=False):
        _, _, H, W = x.shape
        return _random_array(
            self.xp,
            (1, 8, H // self.feat_stride, W // self.feat_stride))


class DummyHead(chainer.Chain):

    def __init__(self, n_class):
        super(DummyHead, self).__init__()
        self.n_class = n_class

    def __call__(self, x, train=False):
        B = x.shape[0]
        bbox_tf = _random_array(self.xp, (B, self.n_class * 4))
        score = _random_array(self.xp, (B, self.n_class))
        return bbox_tf, score


class DummyRegionProposalNetwork(chainer.Chain):

    def __init__(self, n_anchor, n_roi):
        super(DummyRegionProposalNetwork, self).__init__()
        self.n_anchor = n_anchor
        self.n_roi = n_roi

    def __call__(self, x, img_size, scale, train=False):
        B, _, H, W = x.shape
        rpn_bbox_pred = _random_array(
            self.xp, (B, 4 * self.n_anchor, H, W))
        rpn_cls_score = _random_array(
            self.xp, (B, 2 * self.n_anchor, H, W))
        roi_bbox = _generate_bbox(
            self.xp, self.n_roi, img_size[::-1], 16, min(img_size))
        roi = self.xp.concatenate(
            (self.xp.zeros((len(roi_bbox), 1), roi_bbox.dtype), roi_bbox),
            axis=1)
        anchor = _generate_bbox(
            self.xp, self.n_anchor * H * W, img_size[::-1], 16, min(img_size))
        return (chainer.Variable(rpn_bbox_pred),
                chainer.Variable(rpn_cls_score), roi, anchor)


class DummyFasterRCNN(FasterRCNNBase):

    def __init__(self, n_anchor, feat_stride, n_class, n_roi):
        super(DummyFasterRCNN, self).__init__(
            DummyFeature(feat_stride),
            DummyRegionProposalNetwork(n_anchor, n_roi),
            DummyHead(n_class),
            n_class=n_class,
            roi_size=7,
            spatial_scale=1. / feat_stride,
            mean=np.array([[[100, 122.5, 145]]]),
        )


class TestFasterRCNNBase(unittest.TestCase):

    def setUp(self):
        self.n_anchor = 6
        self.feat_stride = 4
        self.n_class = 10
        self.n_roi = 128
        self.link = DummyFasterRCNN(
            n_anchor=self.n_anchor,
            feat_stride=self.feat_stride,
            n_class=self.n_class,
            n_roi=self.n_roi
        )

    def check_call(self):
        xp = self.link.xp

        x1 = chainer.Variable(_random_array(xp, (1, 3, 600, 800)))
        y1 = self.link(x1)
        bbox_tf = y1['bbox_tf']
        score = y1['score']
        self.assertEqual(bbox_tf.shape, (self.n_roi, self.n_class * 4))
        self.assertEqual(score.shape, (self.n_roi, self.n_class))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
