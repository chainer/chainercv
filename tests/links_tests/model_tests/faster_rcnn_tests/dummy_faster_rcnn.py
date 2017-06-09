import numpy as np

import chainer

from chainercv.links.model.faster_rcnn import FasterRCNN
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
        return _random_array(
            self.xp,
            (1, 8, H // self.feat_stride, W // self.feat_stride))


class DummyHead(chainer.Chain):

    def __init__(self, n_class):
        super(DummyHead, self).__init__()
        self.n_class = n_class

    def __call__(self, x, rois, roi_indices):
        n_roi = len(rois)
        cls_locs = chainer.Variable(
            _random_array(self.xp, (n_roi, self.n_class * 4)))
        # For each bbox, the score for a selected class is
        # overwhelmingly higher than the scores for the other classes.
        score_idx = np.random.randint(
            low=0, high=self.n_class, size=(n_roi,))
        scores = self.xp.zeros((n_roi, self.n_class), dtype=np.float32)
        scores[np.arange(n_roi), score_idx] = 100
        scores = chainer.Variable(scores)

        return cls_locs, scores


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


class DummyFasterRCNN(FasterRCNN):

    def __init__(self, n_anchor_base, feat_stride, n_fg_class, n_roi,
                 min_size, max_size
                 ):
        super(DummyFasterRCNN, self).__init__(
            DummyExtractor(feat_stride),
            DummyRegionProposalNetwork(n_anchor_base, n_roi),
            DummyHead(n_fg_class + 1),
            mean=np.array([[[100]], [[122.5]], [[145]]]),
            min_size=min_size,
            max_size=max_size
        )
