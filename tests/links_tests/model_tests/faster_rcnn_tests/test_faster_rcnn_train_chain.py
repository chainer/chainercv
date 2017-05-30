import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainercv.utils import generate_random_bbox

from dummy_faster_rcnn import DummyFasterRCNN


def _random_array(shape):
    return np.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class TestFasterRCNNTrainChain(unittest.TestCase):

    def setUp(self):
        self.n_anchor_base = 6
        self.feat_stride = 4
        self.n_fg_class = 3
        self.n_roi = 24
        self.n_bbox = 3
        self.link = FasterRCNNTrainChain(DummyFasterRCNN(
            n_anchor_base=self.n_anchor_base,
            feat_stride=self.feat_stride,
            n_fg_class=self.n_fg_class,
            n_roi=self.n_roi,
            min_size=600,
            max_size=800,
        ))

        self.bboxes = chainer.Variable(
            generate_random_bbox(self.n_bbox, (600, 800), 16, 350)[np.newaxis])
        _labels = np.random.randint(
            0, self.n_fg_class, size=(1, self.n_bbox)).astype(np.int32)
        self.labels = chainer.Variable(_labels)
        self.imgs = chainer.Variable(_random_array((1, 3, 600, 800)))
        self.scale = chainer.Variable(np.array(1.))

    def check_call(self):
        loss = self.link(self.imgs, self.bboxes, self.labels, self.scale)
        self.assertEqual(loss.shape, ())

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.imgs.to_gpu()
        self.bboxes.to_gpu()
        self.labels.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
