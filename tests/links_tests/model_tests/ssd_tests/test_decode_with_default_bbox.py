import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import decode_with_default_bbox


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


@testing.parameterize(*testing.product({
    'n_fg_class': [1, 5, 20],
    'n_bbox': [1, 20, 100],
    'nms_thresh': [None, 0.5, 1],
    'score_thresh': [0, 0.5, 1, np.inf],
}))
class TestDecodeWithDefaultBbox(unittest.TestCase):

    def _check_decode_with_default_bbox(self, xp):
        default_bbox = _random_array(xp, (self.n_bbox, 4))

        loc = _random_array(xp, (self.n_bbox, 4))
        conf = _random_array(xp, (self.n_bbox, self.n_fg_class + 1))

        bbox, label, score = decode_with_default_bbox(
            loc, conf, default_bbox, (0.1, 0.1),
            self.nms_thresh, self.score_thresh)

        self.assertIsInstance(bbox, xp.ndarray)
        self.assertEqual(bbox.ndim, 2)
        self.assertLessEqual(bbox.shape[0], self.n_bbox * self.n_fg_class)
        self.assertEqual(bbox.shape[1], 4)

        self.assertIsInstance(label, xp.ndarray)
        self.assertEqual(label.ndim, 1)
        self.assertEqual(label.shape[0], bbox.shape[0])

        self.assertIsInstance(score, xp.ndarray)
        self.assertEqual(score.ndim, 1)
        self.assertEqual(score.shape[0], bbox.shape[0])

    def test_decode_with_default_bbox_cpu(self):
        self._check_decode_with_default_bbox(np)

    @attr.gpu
    def test_decode_with_default_bbox_gpu(self):
        self._check_decode_with_default_bbox(chainer.cuda.cupy)


testing.run_module(__name__, __file__)
