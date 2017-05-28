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
