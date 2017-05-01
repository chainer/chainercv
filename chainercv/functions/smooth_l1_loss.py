import chainer.functions as F
import numpy as np
import chainer


# THIS IS PLACED HERE TEMPORARY
def smooth_l1_loss(x, t, inside_weights, outside_weights, sigma):
    # default value is one in the original caffe
    sigma2 = sigma ** 2
    xp = chainer.cuda.get_array_module(x)
    diff = inside_weights * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))

    y = y * outside_weights
    y /= y.shape[0]
    return F.sum(y)
