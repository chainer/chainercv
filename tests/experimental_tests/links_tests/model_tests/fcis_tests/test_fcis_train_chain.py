import numpy as np
import unittest

from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.experimental.links.model.fcis import FCISTrainChain
from chainercv.utils import mask_to_bbox

from tests.experimental_tests.links_tests.model_tests.fcis_tests.test_fcis \
    import _random_array
from tests.experimental_tests.links_tests.model_tests.fcis_tests.test_fcis \
    import DummyFCIS


class TestFCISTrainChain(unittest.TestCase):

    def setUp(self):
        self.n_anchor_base = 6
        self.feat_stride = 4
        self.n_fg_class = 3
        self.n_roi = 24
        self.n_bbox = 3
        self.model = FCISTrainChain(
            DummyFCIS(
                n_anchor_base=self.n_anchor_base,
                feat_stride=self.feat_stride,
                n_fg_class=self.n_fg_class,
                n_roi=self.n_roi,
                roi_size=21,
                min_size=600,
                max_size=1000))

        self.masks = np.random.randint(
            0, 2, size=(1, self.n_bbox, 600, 800)).astype(np.bool)
        self.labels = np.random.randint(
            0, self.n_fg_class, size=(1, self.n_bbox)).astype(np.int32)
        self.imgs = _random_array(np, (1, 3, 600, 800))
        self.scale = np.array(1.)

    def check_call(self, model, imgs, masks, labels, scale):
        bboxes = mask_to_bbox(masks[0])[None]
        loss = model(imgs, masks, labels, bboxes, scale)
        self.assertEqual(loss.shape, ())

    def test_call_cpu(self):
        self.check_call(
            self.model, self.imgs, self.masks, self.labels, self.scale)

    @attr.gpu
    def test_call_gpu(self):
        self.model.to_gpu()
        self.check_call(
            self.model, cuda.to_gpu(self.imgs),
            self.masks, self.labels, self.scale)


testing.run_module(__name__, __file__)
