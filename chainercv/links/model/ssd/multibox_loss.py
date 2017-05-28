import numpy as np

import chainer
import chainer.functions as F


def _elementwise_softmax_cross_entropy(x, t):
    assert x.shape[:-1] == t.shape
    shape = t.shape
    x = F.reshape(x, (-1, shape[-1]))
    t = F.flatten(t)
    return F.reshape(
        F.softmax_cross_entropy(x, t, reduce='no'), shape)


def _hard_negative(x, positive, k):
    xp = chainer.cuda.get_array_module(x, positive)
    x = chainer.cuda.to_cpu(x)
    positive = chainer.cuda.to_cpu(positive)
    rank = (x * (positive - 1)).argsort(axis=1).argsort(axis=1)
    hard_negative = rank < (positive.sum(axis=1) * k)[:, np.newaxis]
    return xp.array(hard_negative)


def multibox_loss(x_loc, x_conf, t_loc, t_conf, k):
    """Computes multibox loss

    This is a loss function used in [#]_.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        x_loc (chainer.Variable): A variable which indicates predicted
            locations of bounding boxes. Its shape is :math:`(B, K, 4)`,
            where :math:`B` is the number of samples in the batch and
            :math:`K` is the number of default bounding boxes.
        x_conf (chainer.Variable): A variable which indicates predicted
            classes of bounding boxes. Its shape is :math:`(B, K, n\_class)`.
            This function assumes the first class is background (negative).
        t_loc (chainer.Variable): A variable which indicates ground truth
            locations of bounding boxes. Its shape is :math:`(B, K, 4)`.
        t_conf (chainer.Variable): A variable which indicates ground truth
            classes of bounding boxes. Its shape is :math:`(B, K)`.
        k (float): A coefficient which is used to hard negative mining.
            This value determines the ratio between the number of positives
            and that of mined negatives. The value used in the original paper
            is :obj:`3`.

    Returns:
        tuple of chainer.Variable:
        This function returns two :obj:`chainer.Variable`: :obj:`loss_loc` and
        :obj:`loss_conf`.
    """
    xp = chainer.cuda.get_array_module(t_conf.data)

    positive = t_conf.data > 0
    if xp.logical_not(positive).all():
        return 0, 0

    loss_loc = F.huber_loss(x_loc, t_loc, 1, reduce='no')
    loss_loc *= positive.astype(loss_loc.dtype)
    loss_loc = F.sum(loss_loc) / positive.sum()

    loss_conf = _elementwise_softmax_cross_entropy(x_conf, t_conf)
    hard_negative = _hard_negative(loss_conf.data, positive, k)
    loss_conf *= xp.logical_or(positive, hard_negative).astype(loss_conf.dtype)
    loss_conf = F.sum(loss_conf) / positive.sum()

    return loss_loc, loss_conf
