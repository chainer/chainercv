from __future__ import division

import numpy as np
import six
import unittest

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.ssd import multibox_loss


@testing.parameterize(*testing.product({
    'k': [3, 10000],
    'batchsize': [1, 5],
    'n_bbox': [10, 500],
    'n_class': [3, 20],
}))
class TestMultiboxLoss(unittest.TestCase):

    def setUp(self):
        self.x_loc = np.random.uniform(
            -10, 10, size=(self.batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.x_conf = np.random.uniform(
            -50, 50, size=(self.batchsize, self.n_bbox, self.n_class)) \
            .astype(np.float32)

        self.t_loc = np.random.uniform(
            -10, 10, size=(self.batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.t_conf = np.random.randint(
            self.n_class, size=(self.batchsize, self.n_bbox)) \
            .astype(np.int32)
        # increase negative samples
        self.t_conf[np.random.uniform(size=self.t_conf.shape) > 0.1] = 0

    def _check_forward(self, x_loc, x_conf, t_loc, t_conf, k):
        x_loc = chainer.Variable(x_loc)
        x_conf = chainer.Variable(x_conf)
        t_loc = chainer.Variable(t_loc)
        t_conf = chainer.Variable(t_conf)

        loc_loss, conf_loss = multibox_loss(
            x_loc, x_conf, t_loc, t_conf, k)

        self.assertIsInstance(loc_loss, chainer.Variable)
        self.assertIsInstance(loc_loss.data, type(x_loc.data))
        self.assertEqual(loc_loss.shape, ())
        self.assertEqual(loc_loss.dtype, x_loc.dtype)

        self.assertIsInstance(conf_loss, chainer.Variable)
        self.assertIsInstance(conf_loss.data, type(x_conf.data))
        self.assertEqual(conf_loss.shape, ())
        self.assertEqual(conf_loss.dtype, x_conf.dtype)

        x_loc = cuda.to_cpu(x_loc.data)
        x_conf = cuda.to_cpu(x_conf.data)
        t_loc = cuda.to_cpu(t_loc.data)
        t_conf = cuda.to_cpu(t_conf.data)
        loc_loss = cuda.to_cpu(loc_loss.data)
        conf_loss = cuda.to_cpu(conf_loss.data)

        n_positive_total = 0
        expect_loc_loss = 0
        expect_conf_loss = 0
        for i in six.moves.xrange(t_conf.shape[0]):
            n_positive = 0
            negatives = list()
            for j in six.moves.xrange(t_conf.shape[1]):
                loc = F.huber_loss(
                    x_loc[np.newaxis, i, j], t_loc[np.newaxis, i, j], 1).data
                conf = F.softmax_cross_entropy(
                    x_conf[np.newaxis, i, j], t_conf[np.newaxis, i, j]).data

                if t_conf[i, j] > 0:
                    n_positive += 1
                    expect_loc_loss += loc
                    expect_conf_loss += conf
                else:
                    negatives.append(conf)

            n_positive_total += n_positive
            if n_positive > 0:
                expect_conf_loss += sum(sorted(negatives)[-n_positive * k:])

        if n_positive_total == 0:
            expect_loc_loss = 0
            expect_conf_loss = 0
        else:
            expect_loc_loss /= n_positive_total
            expect_conf_loss /= n_positive_total

        np.testing.assert_almost_equal(
            loc_loss, expect_loc_loss, decimal=2)
        np.testing.assert_almost_equal(
            conf_loss, expect_conf_loss, decimal=2)

    def test_forward_cpu(self):
        self._check_forward(
            self.x_loc, self.x_conf, self.t_loc, self.t_conf, self.k)

    @attr.gpu
    def test_forward_gpu(self):
        self._check_forward(
            cuda.to_gpu(self.x_loc), cuda.to_gpu(self.x_conf),
            cuda.to_gpu(self.t_loc), cuda.to_gpu(self.t_conf),
            self.k)


testing.run_module(__name__, __file__)
