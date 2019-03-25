import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import Multibox
from chainercv.links.model.ssd import SSD
from chainercv.utils import assert_is_detection_link


def _random_array(xp, shape):
    return xp.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class DummyExtractor(chainer.Link):
    insize = 32
    grids = (10, 4, 1)

    def forward(self, x):
        n_sample = x.shape[0]
        n_dims = (32, 16, 8)
        return [
            chainer.Variable(
                _random_array(self.xp, (n_sample, n_dim, grid, grid)))
            for grid, n_dim in zip(self.grids, n_dims)]


class DummySSD(SSD):

    def __init__(self, n_fg_class):
        super(DummySSD, self).__init__(
            extractor=DummyExtractor(),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2,), (2, 3), (2,))),
            steps=(0.1, 0.25, 1),
            sizes=(0.1, 0.25, 1, 1.2),
            mean=np.array((0, 1, 2)).reshape((-1, 1, 1)))


@testing.parameterize(
    {'n_fg_class': 1},
    {'n_fg_class': 5},
    {'n_fg_class': 20},
)
class TestSSD(unittest.TestCase):

    def setUp(self):
        self.link = DummySSD(n_fg_class=self.n_fg_class)
        self.n_bbox = 10 * 10 * 4 + 4 * 4 * 6 + 1 * 1 * 4

    def _check_call(self):
        x = _random_array(self.link.xp, (1, 3, 32, 32))

        mb_locs, mb_confs = self.link(x)

        self.assertIsInstance(mb_locs, chainer.Variable)
        self.assertIsInstance(mb_locs.array, self.link.xp.ndarray)
        self.assertEqual(mb_locs.shape, (1, self.n_bbox, 4))
        self.assertIsInstance(mb_confs, chainer.Variable)
        self.assertIsInstance(mb_confs.array, self.link.xp.ndarray)
        self.assertEqual(mb_confs.shape, (1, self.n_bbox, self.n_fg_class + 1))

    def test_call_cpu(self):
        self._check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self._check_call()

    def test_prepare(self):
        img = np.random.randint(0, 255, size=(3, 480, 640))
        img = self.link._prepare(img)
        self.assertEqual(img.shape, (3, self.link.insize, self.link.insize))

    def test_use_preset(self):
        self.link.nms_thresh = 0
        self.link.score_thresh = 0

        self.link.use_preset('visualize')
        self.assertEqual(self.link.nms_thresh, 0.45)
        self.assertEqual(self.link.score_thresh, 0.6)

        self.link.nms_thresh = 0
        self.link.score_thresh = 0

        self.link.use_preset('evaluate')
        self.assertEqual(self.link.nms_thresh, 0.45)
        self.assertEqual(self.link.score_thresh, 0.01)

        with self.assertRaises(ValueError):
            self.link.use_preset('unknown')

    def test_predict_cpu(self):
        assert_is_detection_link(self.link, self.n_fg_class)

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_detection_link(self.link, self.n_fg_class)


testing.run_module(__name__, __file__)
