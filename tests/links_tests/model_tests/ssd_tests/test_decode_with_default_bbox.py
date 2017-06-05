import numpy as np
import unittest

from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import decode_with_default_bbox


def _random_array(shape):
    return np.random.uniform(-1, 1, size=shape).astype(np.float32)


@testing.parameterize(*testing.product({
    'n_fg_class': [1, 5, 20],
    'n_bbox': [1, 20, 100],
    'nms_thresh': [None, 0.5, 1],
    'score_thresh': [0, 0.5, 1, np.inf],
}))
class TestDecodeWithDefaultBbox(unittest.TestCase):

    def setUp(self):
        self.loc = _random_array((self.n_bbox, 4))
        self.conf = _random_array((self.n_bbox, self.n_fg_class + 1))
        self.default_bbox = _random_array((self.n_bbox, 4))

    def _check_decode_with_default_bbox(self, loc, conf, default_bbox):
        xp = cuda.get_array_module(loc, conf, default_bbox)

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
        self._check_decode_with_default_bbox(
            self.loc, self.conf, self.default_bbox)

    @attr.gpu
    def test_decode_with_default_bbox_gpu(self):
        self._check_decode_with_default_bbox(
            cuda.to_gpu(self.loc), cuda.to_gpu(self.conf),
            cuda.to_gpu(self.default_bbox))


testing.run_module(__name__, __file__)
