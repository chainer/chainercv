import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links.model.fcis import FCIS
from chainercv.utils import assert_is_instance_segmentation_link
from chainercv.utils import generate_random_bbox


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class DummyExtractor(chainer.Link):

    def __init__(self, feat_stride):
        super(DummyExtractor, self).__init__()
        self.feat_stride = feat_stride

    def __call__(self, x):
        _, _, H, W = x.shape
        rpn_features = _random_array(
            self.xp, (1, 8, H // self.feat_stride, W // self.feat_stride))
        roi_features = _random_array(
            self.xp, (1, 8, H // self.feat_stride, W // self.feat_stride))
        return rpn_features, roi_features


class DummyHead(chainer.Chain):

    def __init__(self, n_class, roi_size):
        super(DummyHead, self).__init__()
        self.n_class = n_class
        self.roi_size = roi_size

    def __call__(self, x, rois, roi_indices, img_size, gt_roi_label=None):
        n_roi = len(rois)
        ag_locs = chainer.Variable(
            _random_array(self.xp, (n_roi, 2, 4)))
        # For each bbox, the score for a selected class is
        # overwhelmingly higher than the scores for the other classes.
        ag_seg_scores = chainer.Variable(
            _random_array(
                self.xp, (n_roi, 2, self.roi_size, self.roi_size)))
        score_idx = np.random.randint(
            low=0, high=self.n_class, size=(n_roi,))
        cls_scores = self.xp.zeros((n_roi, self.n_class), dtype=np.float32)
        cls_scores[np.arange(n_roi), score_idx] = 100
        cls_scores = chainer.Variable(cls_scores)

        return ag_seg_scores, ag_locs, cls_scores, rois, roi_indices


class DummyRegionProposalNetwork(chainer.Chain):

    def __init__(self, n_anchor_base, n_roi):
        super(DummyRegionProposalNetwork, self).__init__()
        self.n_anchor_base = n_anchor_base
        self.n_roi = n_roi

    def __call__(self, x, img_size, scale):
        B, _, H, W = x.shape
        n_anchor = self.n_anchor_base * H * W

        rpn_locs = _random_array(self.xp, (B, n_anchor, 4))
        rpn_cls_scores = _random_array(self.xp, (B, n_anchor, 2))
        rois = self.xp.asarray(generate_random_bbox(
            self.n_roi, img_size, 16, min(img_size)))
        roi_indices = self.xp.zeros((len(rois),), dtype=np.int32)
        anchor = self.xp.asarray(generate_random_bbox(
            n_anchor, img_size, 16, min(img_size)))
        return (chainer.Variable(rpn_locs),
                chainer.Variable(rpn_cls_scores), rois, roi_indices, anchor)


class DummyFCIS(FCIS):

    def __init__(
        self, n_anchor_base, feat_stride,
        n_fg_class, n_roi, roi_size,
        min_size, max_size
    ):

        super(DummyFCIS, self).__init__(
            DummyExtractor(feat_stride),
            DummyRegionProposalNetwork(n_anchor_base, n_roi),
            DummyHead(n_fg_class + 1, roi_size),
            mean=np.array([[[123.15]], [[115.90]], [[103.06]]]),
            min_size=min_size,
            max_size=max_size,
            loc_normalize_mean=(0.0, 0.0, 0.0, 0.0),
            loc_normalize_std=(0.2, 0.2, 0.5, 0.5))


class TestFCIS(unittest.TestCase):

    def setUp(self):
        self.n_anchor_base = 6
        self.feat_stride = 4
        n_fg_class = 4
        self.n_class = n_fg_class + 1
        self.n_roi = 24
        self.roi_size = 21
        self.link = DummyFCIS(
            n_anchor_base=self.n_anchor_base,
            feat_stride=self.feat_stride,
            n_fg_class=n_fg_class,
            n_roi=self.n_roi,
            roi_size=21,
            min_size=600,
            max_size=1000,
        )

    def check_call(self):
        xp = self.link.xp

        x1 = chainer.Variable(_random_array(xp, (1, 3, 600, 800)))
        roi_ag_seg_scores, roi_ag_locs, roi_cls_scores, rois, roi_indices = \
            self.link(x1)

        self.assertIsInstance(roi_ag_seg_scores, chainer.Variable)
        self.assertIsInstance(roi_ag_seg_scores.array, xp.ndarray)
        self.assertEqual(
            roi_ag_seg_scores.shape,
            (self.n_roi, 2, self.roi_size, self.roi_size))

        self.assertIsInstance(roi_ag_locs, chainer.Variable)
        self.assertIsInstance(roi_ag_locs.array, xp.ndarray)
        self.assertEqual(roi_ag_locs.shape, (self.n_roi, 2, 4))

        self.assertIsInstance(roi_cls_scores, chainer.Variable)
        self.assertIsInstance(roi_cls_scores.array, xp.ndarray)
        self.assertEqual(roi_cls_scores.shape, (self.n_roi, self.n_class))

        self.assertIsInstance(rois, xp.ndarray)
        self.assertEqual(rois.shape, (self.n_roi, 4))

        self.assertIsInstance(roi_indices, xp.ndarray)
        self.assertEqual(roi_indices.shape, (self.n_roi,))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def test_predict_cpu(self):
        assert_is_instance_segmentation_link(self.link, self.n_class - 1)

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_instance_segmentation_link(self.link, self.n_class - 1)


@testing.parameterize(
    {'in_shape': (3, 100, 100), 'expected_shape': (3, 200, 200)},
    {'in_shape': (3, 200, 50), 'expected_shape': (3, 400, 100)},
    {'in_shape': (3, 400, 100), 'expected_shape': (3, 400, 100)},
    {'in_shape': (3, 300, 600), 'expected_shape': (3, 200, 400)},
    {'in_shape': (3, 600, 900), 'expected_shape': (3, 200, 300)}
)
class TestFCISPrepare(unittest.TestCase):

    min_size = 200
    max_size = 400

    def setUp(self):
        self.link = DummyFCIS(
            n_anchor_base=1,
            feat_stride=16,
            n_fg_class=20,
            n_roi=1,
            roi_size=21,
            min_size=self.min_size,
            max_size=self.max_size
        )

    def check_prepare(self):
        x = _random_array(np, self.in_shape)
        out = self.link.prepare(x)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.expected_shape)

    def test_prepare_cpu(self):
        self.check_prepare()

    @attr.gpu
    def test_prepare_gpu(self):
        self.link.to_gpu()
        self.check_prepare()


testing.run_module(__name__, __file__)
