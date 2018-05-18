import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import PSPNet
from chainercv.utils import assert_is_semantic_segmentation_link


@testing.parameterize(
    {'train': False, 'mid_stride': False},
    {'train': True, 'mid_stride': True}
)
@attr.slow
class TestPSPNet(unittest.TestCase):

    def setUp(self):
        self.n_class = 10
        self.n_blocks = [3, 4, 23, 3]
        self.input_size = (128, 160)
        self.pyramids = [6, 3, 2, 1]
        self.xsize = (2, 3, 128, 160)
        self.link = PSPNet(
            n_class=self.n_class, input_size=self.input_size,
            n_blocks=self.n_blocks, pyramids=self.pyramids,
            mid_stride=self.mid_stride)

    def test_ppm(self):
        ksizes = self.link.ppm.ksizes
        self.assertEqual(ksizes, [(2, 3), (5, 6), (8, 10), (16, 20)])

    def check_call(self):
        with chainer.using_config('train', self.train):
            xp = self.link.xp
            x = chainer.Variable(xp.random.uniform(
                low=-1, high=1, size=self.xsize).astype(np.float32))
            ys = self.link(x)
            if self.train:
                self.assertEqual(len(ys), 2)
                aux, y = ys
                self.assertIsInstance(aux, chainer.Variable)
                self.assertIsInstance(aux.data, xp.ndarray)
                self.assertEqual(aux.shape, (2, self.n_class, 128, 160))
            else:
                self.assertTrue(isinstance(ys, chainer.Variable))
                y = ys

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, xp.ndarray)
        self.assertEqual(y.shape, (2, self.n_class, 128, 160))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def test_predict_cpu(self):
        assert_is_semantic_segmentation_link(self.link, self.n_class)

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        assert_is_semantic_segmentation_link(self.link, self.n_class)


testing.run_module(__name__, __file__)
