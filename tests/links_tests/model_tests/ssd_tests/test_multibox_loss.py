import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import multibox_loss


@testing.parameterize(*testing.product({
    'k': [3, 1000],
    'batchsize': [1, 2],
    'n_bbox': [5, 10],
    'n_class': [3, 5],
}))
class TestMultiboxLoss(unittest.TestCase):

    def setUp(self):
        self.x_loc = chainer.Variable(np.random.uniform(
            -1, 1, size=(self.batchsize, self.n_bbox, 4))
            .astype(np.float32))
        self.x_conf = chainer.Variable(np.random.uniform(
            -1, 1, size=(self.batchsize, self.n_bbox, self.n_class))
            .astype(np.float32))

        self.t_loc = chainer.Variable(np.random.uniform(
            -1, 1, size=(self.batchsize, self.n_bbox, 4))
            .astype(np.float32))
        self.t_conf = chainer.Variable(np.random.randint(
            self.n_class, size=(self.batchsize, self.n_bbox))
            .astype(np.int32))

    def _check_forward(self, x_loc, x_conf, t_loc, t_conf):
        loss_loc, loss_conf = multibox_loss(
            x_loc, x_conf, t_loc, t_conf, self.k)

        self.assertIsInstance(loss_loc, chainer.Variable)
        self.assertIsInstance(loss_loc.data, type(x_loc.data))
        self.assertEqual(loss_loc.shape, ())
        self.assertEqual(loss_loc.dtype, x_loc.dtype)

        self.assertIsInstance(loss_conf, chainer.Variable)
        self.assertIsInstance(loss_conf.data, type(x_conf.data))
        self.assertEqual(loss_conf.shape, ())
        self.assertEqual(loss_conf.dtype, x_conf.dtype)

    def test_forward_cpu(self):
        self._check_forward(
            self.x_loc, self.x_conf, self.t_loc, self.t_conf)

    @attr.gpu
    def test_forward_gpu(self):
        self.x_loc.to_gpu()
        self.x_conf.to_gpu()
        self.t_loc.to_gpu()
        self.t_conf.to_gpu()
        self._check_forward(
            self.x_loc, self.x_conf, self.t_loc, self.t_conf)


testing.run_module(__name__, __file__)
