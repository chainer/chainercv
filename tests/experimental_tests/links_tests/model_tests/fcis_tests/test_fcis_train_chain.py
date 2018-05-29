import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links.model.fcis import FCISTrainChain

from test_fcis import DummyFCIS
from test_fcis import _random_array


class TestFasterRCNNTrainChain(unittest.TestCase):

    def setUp(self):
        self.n_anchor_base = 6
        self.feat_stride = 4
        self.n_fg_class = 3
        self.n_roi = 24
        self.n_bbox = 3
        self.link = FCISTrainChain(
            DummyFCIS(
                n_anchor_base=self.n_anchor_base,
                feat_stride=self.feat_stride,
                n_fg_class=self.n_fg_class,
                n_roi=self.n_roi,
                roi_size=21,
                min_size=600,
                max_size=1000))

        _masks = np.random.randint(
            0, 2, size=(1, self.n_bbox, 600, 800)).astype(np.bool)
        _labels = np.random.randint(
            0, self.n_fg_class, size=(1, self.n_bbox)).astype(np.int32)
        self.masks = chainer.Variable(_masks)
        self.labels = chainer.Variable(_labels)
        self.imgs = chainer.Variable(_random_array((1, 3, 600, 800)))
        self.scale = chainer.Variable(np.array(1.))

    def check_call(self):
        loss = self.link(self.imgs, self.masks, self.labels, self.scale)
        self.assertEqual(loss.shape, ())

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.imgs.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)
