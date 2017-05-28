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
        self.x_locs = np.random.uniform(
            -10, 10, size=(self.batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.x_confs = np.random.uniform(
            -50, 50, size=(self.batchsize, self.n_bbox, self.n_class)) \
            .astype(np.float32)

        self.t_locs = np.random.uniform(
            -10, 10, size=(self.batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.t_confs = np.random.randint(
            self.n_class, size=(self.batchsize, self.n_bbox)) \
            .astype(np.int32)
        # increase negative samples
        self.t_confs[np.random.uniform(size=self.t_confs.shape) > 0.1] = 0

    def _check_forward(self, x_locs, x_confs, t_locs, t_confs, k):
        x_locs = chainer.Variable(x_locs)
        x_confs = chainer.Variable(x_confs)
        t_locs = chainer.Variable(t_locs)
        t_confs = chainer.Variable(t_confs)

        loc_loss, conf_loss = multibox_loss(
            x_locs, x_confs, t_locs, t_confs, k)

        self.assertIsInstance(loc_loss, chainer.Variable)
        self.assertIsInstance(loc_loss.data, type(x_locs.data))
        self.assertEqual(loc_loss.shape, ())
        self.assertEqual(loc_loss.dtype, x_locs.dtype)

        self.assertIsInstance(conf_loss, chainer.Variable)
        self.assertIsInstance(conf_loss.data, type(x_confs.data))
        self.assertEqual(conf_loss.shape, ())
        self.assertEqual(conf_loss.dtype, x_confs.dtype)

        x_locs = cuda.to_cpu(x_locs.data)
        x_confs = cuda.to_cpu(x_confs.data)
        t_locs = cuda.to_cpu(t_locs.data)
        t_confs = cuda.to_cpu(t_confs.data)
        loc_loss = cuda.to_cpu(loc_loss.data)
        conf_loss = cuda.to_cpu(conf_loss.data)

        n_positive_total = 0
        expect_loc_loss = 0
        expect_conf_loss = 0
        for i in six.moves.xrange(t_confs.shape[0]):
            n_positive = 0
            negatives = list()
            for j in six.moves.xrange(t_confs.shape[1]):
                loc = F.huber_loss(
                    x_locs[np.newaxis, i, j], t_locs[np.newaxis, i, j], 1).data
                conf = F.softmax_cross_entropy(
                    x_confs[np.newaxis, i, j], t_confs[np.newaxis, i, j]).data

                if t_confs[i, j] > 0:
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
            self.x_locs, self.x_confs, self.t_locs, self.t_confs, self.k)

    @attr.gpu
    def test_forward_gpu(self):
        self._check_forward(
            cuda.to_gpu(self.x_locs), cuda.to_gpu(self.x_confs),
            cuda.to_gpu(self.t_locs), cuda.to_gpu(self.t_confs),
            self.k)


testing.run_module(__name__, __file__)
