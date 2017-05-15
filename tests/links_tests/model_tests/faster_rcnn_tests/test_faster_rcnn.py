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

    def __call__(self, x, rois, batch_indices, train=False):
        bbox_tfs = []
        scores = []
        n_roi = len(rois)
        bbox_tfs = chainer.Variable(
            _random_array(self.xp, (n_roi, self.n_class * 4)))
        # For each bbox, the score for a selected class is
        # overwhelmingly higher than the scores for the other classes.
        score_idx = np.random.randint(
            low=0, high=self.n_class, size=(n_roi,))
        scores = self.xp.zeros((n_roi, self.n_class), dtype=np.float32)
        scores[np.arange(n_roi), score_idx] = 100
        scores = chainer.Variable(scores)

        return bbox_tfs, scores


class DummyRegionProposalNetwork(chainer.Chain):

    def __init__(self, n_anchor, n_roi):
        super(DummyRegionProposalNetwork, self).__init__()
        self.n_anchor = n_anchor
        self.n_roi = n_roi

    def __call__(self, x, img_size, scale, train=False):
        B, _, H, W = x.shape
        rpn_bbox_preds = _random_array(
            self.xp, (B, 4 * self.n_anchor, H, W))
        rpn_cls_scores = _random_array(
            self.xp, (B, 2 * self.n_anchor, H, W))
        rois = _generate_bbox(
            self.xp, self.n_roi, img_size[::-1], 16, min(img_size))
        batch_indices = self.xp.zeros((len(rois),), dtype=np.int32)
        anchor = _generate_bbox(
            self.xp, self.n_anchor * H * W, img_size[::-1], 16, min(img_size))
        return (chainer.Variable(rpn_bbox_preds),
                chainer.Variable(rpn_cls_scores), rois, batch_indices, anchor)


class DummyFasterRCNN(FasterRCNNBase):

    def __init__(self, n_anchor, feat_stride, n_class, n_roi):
        super(DummyFasterRCNN, self).__init__(
            DummyFeature(feat_stride),
            DummyRegionProposalNetwork(n_anchor, n_roi),
            DummyHead(n_class),
            n_class=n_class,
            mean=np.array([[[100]], [[122.5]], [[145]]]),
        )


class TestFasterRCNNBase(unittest.TestCase):

    def setUp(self):
        self.n_anchor = 6
        self.feat_stride = 4
        self.n_class = 4
        self.n_roi = 24
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
        roi_bboxes = y1['roi_bboxes']
        roi_scores = y1['roi_scores']
        self.assertEqual(roi_bboxes.shape, (self.n_roi, self.n_class * 4))
        self.assertEqual(roi_scores.shape, (self.n_roi, self.n_class))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def check_predict(self):
        imgs = [
            _random_array(np, (3, 640, 480)),
            _random_array(np, (3, 320, 320))]

        bboxes, labels, scores = self.link.predict(imgs)

        self.assertEqual(len(bboxes), len(imgs))
        self.assertEqual(len(labels), len(imgs))
        self.assertEqual(len(scores), len(imgs))

        for bbox, label, score in zip(bboxes, labels, scores):
            self.assertIsInstance(bbox, np.ndarray)
            self.assertEqual(bbox.dtype, np.float32)
            self.assertEqual(bbox.ndim, 2)
            self.assertLessEqual(bbox.shape[0], self.n_roi)
            self.assertEqual(bbox.shape[1], 4)

            self.assertIsInstance(label, np.ndarray)
            self.assertEqual(label.dtype, np.int32)
            self.assertEqual(label.shape, (bbox.shape[0],))

            self.assertIsInstance(score, np.ndarray)
            self.assertEqual(score.dtype, np.float32)
            self.assertEqual(score.shape, (bbox.shape[0],))

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()


testing.run_module(__name__, __file__)
