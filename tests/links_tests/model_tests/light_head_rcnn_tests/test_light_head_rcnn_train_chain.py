import numpy as np
import unittest

from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.light_head_rcnn import LightHeadRCNNTrainChain
from chainercv.utils import generate_random_bbox

from tests.links_tests.model_tests.light_head_rcnn_tests.test_light_head_rcnn \
    import DummyLightHeadRCNN


def _random_array(shape):
    return np.array(
        np.random.uniform(-1, 1, size=shape), dtype=np.float32)


class TestLightHeadRCNNTrainChain(unittest.TestCase):

    def setUp(self):
        self.n_anchor_base = 6
        self.feat_stride = 4
        self.n_fg_class = 3
        self.n_roi = 24
        self.n_bbox = 3
        self.model = LightHeadRCNNTrainChain(
            DummyLightHeadRCNN(
                n_anchor_base=self.n_anchor_base,
                feat_stride=self.feat_stride,
                n_fg_class=self.n_fg_class,
                n_roi=self.n_roi,
                min_size=600,
                max_size=800,
                loc_normalize_mean=(0., 0., 0., 0.),
                loc_normalize_std=(0.1, 0.1, 0.2, 0.2),))

        self.bboxes = generate_random_bbox(
            self.n_bbox, (600, 800), 16, 350)[np.newaxis]
        self.labels = np.random.randint(
            0, self.n_fg_class, size=(1, self.n_bbox)).astype(np.int32)
        self.imgs = _random_array((1, 3, 600, 800))
        self.scales = np.array([1.])

    def check_call(self, model, imgs, bboxes, labels, scales):
        loss = self.model(imgs, bboxes, labels, scales)
        self.assertEqual(loss.shape, ())

    def test_call_cpu(self):
        self.check_call(
            self.model, self.imgs, self.bboxes, self.labels, self.scales)

    @attr.gpu
    def test_call_gpu(self):
        self.model.to_gpu()
        self.check_call(
            self.model, cuda.to_gpu(self.imgs),
            self.bboxes, self.labels, self.scales)


testing.run_module(__name__, __file__)
