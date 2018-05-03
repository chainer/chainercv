import mock
import numpy as np
import unittest

import chainer
from chainer.backends.cuda import to_cpu
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.yolo import YOLOBase
from chainercv.utils import assert_is_detection_link
from chainercv.utils import generate_random_bbox


class DummyYOLO(YOLOBase):

    _insize = 64
    _n_anchor = 16
    _n_fg_class = 20

    def __init__(self):
        super(DummyYOLO, self).__init__()

        self.extractor = mock.Mock()
        self.extractor.insize = self._insize

    def __call__(self, x):
        assert(x.shape[1:] == (3, self._insize, self._insize))
        self._value = self.xp.random.uniform(
            size=(x.shape[0], self._n_anchor, 4 + 1 + self._n_fg_class)) \
            .astype(np.float32)
        return chainer.Variable(self._value)

    def _decode(self, loc, conf):
        value = to_cpu(self._value)
        loc = to_cpu(loc)
        conf = to_cpu(conf)

        if not hasattr(self, '_count'):
            self._count = 0
        np.testing.assert_equal(loc, value[self._count, :, :4])
        np.testing.assert_equal(conf, value[self._count, :, 4:])
        self._count += 1

        n_bbox = np.random.randint(self._n_anchor - 1)
        bbox = generate_random_bbox(
            n_bbox, (self._insize, self._insize), 8, 48)
        label = np.random.randint(self._n_fg_class - 1, size=n_bbox) \
            .astype(np.int32)
        score = np.random.uniform(size=n_bbox).astype(np.float32)
        return bbox, label, score


class TestYOLOBase(unittest.TestCase):

    def setUp(self):
        self.link = DummyYOLO()

    def test_use_preset(self):
        self.link.nms_thresh = 0
        self.link.score_thresh = 0

        self.link.use_preset('visualize')
        self.assertEqual(self.link.nms_thresh, 0.45)
        self.assertEqual(self.link.score_thresh, 0.5)

        self.link.nms_thresh = 0
        self.link.score_thresh = 0

        self.link.use_preset('evaluate')
        self.assertEqual(self.link.nms_thresh, 0.45)
        self.assertEqual(self.link.score_thresh, 0.005)

        with self.assertRaises(ValueError):
            self.link.use_preset('unknown')

    def test_predict_cpu(self):
        assert_is_detection_link(self.link, self.link._n_fg_class)

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_detection_link(self.link, self.link._n_fg_class)


testing.run_module(__name__, __file__)
