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


try:
    from chainermn import create_communicator
    _chainermn_available = True
except ImportError:
    _chainermn_available = False


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

    def _check_forward(
            self, mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k, comm=None):
        if self.variable:
            mb_locs = chainer.Variable(mb_locs)
            mb_confs = chainer.Variable(mb_confs)
            gt_mb_locs = chainer.Variable(gt_mb_locs)
            gt_mb_labels = chainer.Variable(gt_mb_labels)

        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k, comm)

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

    @unittest.skipIf(not _chainermn_available, 'ChainerMN is not installed')
    def test_multi_node_forward_cpu(self):
        self._check_forward(
            self.mb_locs, self.mb_confs,
            self.gt_mb_locs, self.gt_mb_labels,
            self.k, create_communicator('naive'))

    @unittest.skipIf(not _chainermn_available, 'ChainerMN is not installed')
    @attr.gpu
    def test_multi_node_forward_gpu(self):
        self._check_forward(
            cuda.to_gpu(self.mb_locs), cuda.to_gpu(self.mb_confs),
            cuda.to_gpu(self.gt_mb_locs), cuda.to_gpu(self.gt_mb_labels),
            self.k, create_communicator('naive'))


testing.run_module(__name__, __file__)
