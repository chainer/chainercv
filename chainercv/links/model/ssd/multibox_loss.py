from __future__ import division

import numpy as np

import chainer
import chainer.functions as F


def _elementwise_softmax_cross_entropy(x, t):
    assert x.shape[:-1] == t.shape
    shape = t.shape
    x = F.reshape(x, (-1, x.shape[-1]))
    t = F.flatten(t)
    return F.reshape(
        F.softmax_cross_entropy(x, t, reduce='no'), shape)


def _hard_negative(x, positive, k):
    rank = (x * (positive - 1)).argsort(axis=1).argsort(axis=1)
    hard_negative = rank < (positive.sum(axis=1) * k)[:, np.newaxis]
    return hard_negative


def multibox_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k, comm=None):
    """Computes multibox losses.

    This is a loss function used in [#]_.
    This function returns :obj:`loc_loss` and :obj:`conf_loss`.
    :obj:`loc_loss` is a loss for localization and
    :obj:`conf_loss` is a loss for classification.
    The formulas of these losses can be found in
    the equation (2) and (3) in the original paper.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        mb_locs (chainer.Variable or array): The offsets and scales
            for predicted bounding boxes.
            Its shape is :math:`(B, K, 4)`,
            where :math:`B` is the number of samples in the batch and
            :math:`K` is the number of default bounding boxes.
        mb_confs (chainer.Variable or array): The classes of predicted
            bounding boxes.
            Its shape is :math:`(B, K, n\_class)`.
            This function assumes the first class is background (negative).
        gt_mb_locs (chainer.Variable or array): The offsets and scales
            for ground truth bounding boxes.
            Its shape is :math:`(B, K, 4)`.
        gt_mb_labels (chainer.Variable or array): The classes of ground truth
            bounding boxes.
            Its shape is :math:`(B, K)`.
        k (float): A coefficient which is used for hard negative mining.
            This value determines the ratio between the number of positives
            and that of mined negatives. The value used in the original paper
            is :obj:`3`.
        comm (~chainermn.communicators.CommunicatorBase):
            A ChainerMN communicator.
            If it is specified, the number of positive examples is computed
            among all GPUs.

    Returns:
        tuple of chainer.Variable:
        This function returns two :obj:`chainer.Variable`: :obj:`loc_loss` and
        :obj:`conf_loss`.
    """
    mb_locs = chainer.as_variable(mb_locs)
    mb_confs = chainer.as_variable(mb_confs)
    gt_mb_locs = chainer.as_variable(gt_mb_locs)
    gt_mb_labels = chainer.as_variable(gt_mb_labels)

    xp = chainer.backends.cuda.get_array_module(gt_mb_labels.array)
    with chainer.backends.cuda.get_device_from_array(gt_mb_labels.array):
        positive = gt_mb_labels.array > 0
        n_positive = positive.sum()

        if comm:
            n_positive = comm.allreduce_obj(n_positive) / comm.size

        if n_positive == 0:
            z = chainer.Variable(xp.zeros((), dtype=np.float32))
            return z, z

        loc_loss = F.huber_loss(mb_locs, gt_mb_locs, 1, reduce='no')
        loc_loss = F.sum(loc_loss, axis=-1)
        loc_loss *= positive.astype(loc_loss.dtype)
        loc_loss = F.sum(loc_loss) / n_positive

        conf_loss = _elementwise_softmax_cross_entropy(mb_confs, gt_mb_labels)
        hard_negative = _hard_negative(conf_loss.array, positive, k)
        conf_loss *= xp.logical_or(
            positive, hard_negative).astype(conf_loss.dtype)
        conf_loss = F.sum(conf_loss) / n_positive

    return loc_loss, conf_loss
