from __future__ import division

import numpy as np
import six
import unittest

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing
from chainermn import create_communicator

from chainercv.links.model.ssd import multibox_loss
from chainercv.utils.testing import attr


@testing.parameterize(*testing.product({
    'k': [3, 10000],
    'batchsize': [1, 5],
    'n_bbox': [10, 500],
    'n_class': [3, 20],
    'variable': [True, False],
}))
class TestMultiboxLoss(unittest.TestCase):

    def setUp(self):
        self.mb_locs = np.random.uniform(
            -10, 10, size=(self.batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.mb_confs = np.random.uniform(
            -50, 50, size=(self.batchsize, self.n_bbox, self.n_class)) \
            .astype(np.float32)

        self.gt_mb_locs = np.random.uniform(
            -10, 10, size=(self.batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.gt_mb_labels = np.random.randint(
            self.n_class, size=(self.batchsize, self.n_bbox)) \
            .astype(np.int32)
        # increase negative samples
        self.gt_mb_labels[np.random.uniform(
            size=self.gt_mb_labels.shape) > 0.1] = 0

    def _check_forward(self, mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k):
        if self.variable:
            mb_locs = chainer.Variable(mb_locs)
            mb_confs = chainer.Variable(mb_confs)
            gt_mb_locs = chainer.Variable(gt_mb_locs)
            gt_mb_labels = chainer.Variable(gt_mb_labels)

        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k)

        self.assertIsInstance(loc_loss, chainer.Variable)
        self.assertEqual(loc_loss.shape, ())
        self.assertEqual(loc_loss.dtype, mb_locs.dtype)

        self.assertIsInstance(conf_loss, chainer.Variable)
        self.assertEqual(conf_loss.shape, ())
        self.assertEqual(conf_loss.dtype, mb_confs.dtype)

        if self.variable:
            mb_locs = mb_locs.array
            mb_confs = mb_confs.array
            gt_mb_locs = gt_mb_locs.array
            gt_mb_labels = gt_mb_labels.array

        mb_locs = cuda.to_cpu(mb_locs)
        mb_confs = cuda.to_cpu(mb_confs)
        gt_mb_locs = cuda.to_cpu(gt_mb_locs)
        gt_mb_labels = cuda.to_cpu(gt_mb_labels)
        loc_loss = cuda.to_cpu(loc_loss.array)
        conf_loss = cuda.to_cpu(conf_loss.array)

        n_positive_total = 0
        expect_loc_loss = 0
        expect_conf_loss = 0
        for i in six.moves.xrange(gt_mb_labels.shape[0]):
            n_positive = 0
            negatives = []
            for j in six.moves.xrange(gt_mb_labels.shape[1]):
                loc = F.huber_loss(
                    mb_locs[np.newaxis, i, j],
                    gt_mb_locs[np.newaxis, i, j], 1).array
                conf = F.softmax_cross_entropy(
                    mb_confs[np.newaxis, i, j],
                    gt_mb_labels[np.newaxis, i, j]).array

                if gt_mb_labels[i, j] > 0:
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
            self.mb_locs, self.mb_confs,
            self.gt_mb_locs, self.gt_mb_labels,
            self.k)

    @attr.gpu
    def test_forward_gpu(self):
        self._check_forward(
            cuda.to_gpu(self.mb_locs), cuda.to_gpu(self.mb_confs),
            cuda.to_gpu(self.gt_mb_locs), cuda.to_gpu(self.gt_mb_labels),
            self.k)


@attr.mpi
class TestMultiNodeMultiboxLoss(unittest.TestCase):

    k = 3
    batchsize = 5
    n_bbox = 10
    n_class = 3

    def setUp(self):
        self.comm = create_communicator('naive')
        batchsize = self.comm.size * self.batchsize

        np.random.seed(0)
        self.mb_locs = np.random.uniform(
            -10, 10, size=(batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.mb_confs = np.random.uniform(
            -50, 50, size=(batchsize, self.n_bbox, self.n_class)) \
            .astype(np.float32)
        self.gt_mb_locs = np.random.uniform(
            -10, 10, size=(batchsize, self.n_bbox, 4)) \
            .astype(np.float32)
        self.gt_mb_labels = np.random.randint(
            self.n_class, size=(batchsize, self.n_bbox)) \
            .astype(np.int32)

        self.mb_locs_local = self.comm.mpi_comm.scatter(
            self.mb_locs.reshape(
                (self.comm.size, self.batchsize, self.n_bbox, 4)))
        self.mb_confs_local = self.comm.mpi_comm.scatter(
            self.mb_confs.reshape(
                (self.comm.size, self.batchsize, self.n_bbox, self.n_class)))
        self.gt_mb_locs_local = self.comm.mpi_comm.scatter(
            self.gt_mb_locs.reshape(
                (self.comm.size, self.batchsize, self.n_bbox, 4)))
        self.gt_mb_labels_local = self.comm.mpi_comm.scatter(
            self.gt_mb_labels.reshape(
                (self.comm.size, self.batchsize, self.n_bbox)))

    def _check_forward(
            self, mb_locs_local, mb_confs_local,
            gt_mb_locs_local, gt_mb_labels_local, k):
        loc_loss_local, conf_loss_local = multibox_loss(
            mb_locs_local, mb_confs_local,
            gt_mb_locs_local, gt_mb_labels_local, k, self.comm)

        loc_loss_local = cuda.to_cpu(loc_loss_local.array)
        conf_loss_local = cuda.to_cpu(conf_loss_local.array)
        loc_loss = self.comm.allreduce_obj(loc_loss_local) / self.comm.size
        conf_loss = self.comm.allreduce_obj(conf_loss_local) / self.comm.size

        expect_loc_loss, expect_conf_loss = multibox_loss(
            self.mb_locs, self.mb_confs, self.gt_mb_locs, self.gt_mb_labels, k)
        np.testing.assert_almost_equal(
            loc_loss, expect_loc_loss.array, decimal=2)
        np.testing.assert_almost_equal(
            conf_loss, expect_conf_loss.array, decimal=2)

    def test_multi_node_forward_cpu(self):
        self._check_forward(
            self.mb_locs, self.mb_confs,
            self.gt_mb_locs, self.gt_mb_labels,
            self.k)

    @attr.gpu
    def test_multi_node_forward_gpu(self):
        self._check_forward(
            cuda.to_gpu(self.mb_locs), cuda.to_gpu(self.mb_confs),
            cuda.to_gpu(self.gt_mb_locs), cuda.to_gpu(self.gt_mb_labels),
            self.k)


testing.run_module(__name__, __file__)
