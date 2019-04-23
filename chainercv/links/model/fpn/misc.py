from __future__ import division

import numpy as np

from chainer.backends import cuda
import chainer.functions as F

from chainercv import transforms


exp_clip = np.log(1000 / 16)


def smooth_l1(x, t, beta):
    return F.huber_loss(x, t, beta, reduce='no') / beta


# to avoid out of memory
def argsort(x):
    xp = cuda.get_array_module(x)
    i = np.argsort(cuda.to_cpu(x))
    if xp is np:
        return i
    else:
        return cuda.to_gpu(i)


# to avoid out of memory
def choice(x, size):
    xp = cuda.get_array_module(x)
    y = np.random.choice(cuda.to_cpu(x), size, replace=False)
    if xp is np:
        return y
    else:
        return cuda.to_gpu(y)


def scale_img(img, min_size, max_size):
    """Process image."""
    _, H, W = img.shape
    scale = min_size / min(H, W)
    if scale * max(H, W) > max_size:
        scale = max_size / max(H, W)
    H, W = int(H * scale), int(W * scale)
    img = transforms.resize(img, (H, W))
    return img, scale
