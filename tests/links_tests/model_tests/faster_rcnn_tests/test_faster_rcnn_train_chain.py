import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain

from dummy_faster_rcnn import DummyFasterRCNN


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


class TestFasterRCNNTrainChain(unittest.TestCase):

    def setUp(self):
        self.n_anchor_base = 6
        self.feat_stride = 4
        self.n_fg_class = 3
        self.n_roi = 24
        self.link = FasterRCNNTrainChain(DummyFasterRCNN(
            n_anchor_base=self.n_anchor_base,
            feat_stride=self.feat_stride,
            n_fg_class=self.n_fg_class,
            n_roi=self.n_roi,
            min_size=600,
            max_size=800,
        ))

    def check_call(self):
        xp = self.link.xp

        n_bbox = 3
        imgs = chainer.Variable(_random_array(xp, (1, 3, 600, 800)))
        bboxes = chainer.Variable(
            _generate_bbox(xp, n_bbox, (600, 800), 16, 350)[np.newaxis])
        labels = xp.random.randint(0, self.n_fg_class + 1, size=(n_bbox,))
        labels = chainer.Variable(labels[np.newaxis].astype(np.int32))
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
