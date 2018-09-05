import chainer.functions as F


def smooth_l1(x, t, beta):
    return F.huber_loss(x, t, beta, reduce='no') / beta
