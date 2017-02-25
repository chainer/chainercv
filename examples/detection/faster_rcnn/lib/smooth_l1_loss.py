from chainer import cuda
from chainer import function
from chainer.utils import type_check

import numpy


class SmoothL1Loss(function.Function):

    def __init__(self, sigma, inside_weights, outside_weights):
        self.sigma2 = sigma * sigma
        self.inside_weights = inside_weights
        self.outside_weights = outside_weights

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1 = inputs
        self.diff = self.inside_weights * (x0 - x1)
        abs_diff = xp.abs(self.diff)
        flag = abs_diff < 1.0 / self.sigma2
        y = (flag * 0.5 * xp.square(self.diff) * self.sigma2 +
             (~flag) * (abs_diff - 0.5 / self.sigma2))
        if xp == cuda.cupy:
            with cuda.Device(cuda.get_device(y)):
                num = xp.prod(xp.asarray(y.shape))
        else:
            num = xp.prod(y.shape)
        return xp.array(y.sum() / num).astype(numpy.float32),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        mask = xp.abs(self.diff) < 1.0 / self.sigma2
        gx = gy[0].reshape(gy[0].shape + (1,) * (self.diff.ndim - 1)) * \
            xp.where(mask, self.sigma2 * self.diff, xp.sign(self.diff))
        gx *= self.inside_weights
        gx *= self.outside_weights
        return gx, -gx


def smooth_l1_loss(x, t, inside_weights, outside_weights, sigma):
    return SmoothL1Loss(sigma=sigma, inside_weights=inside_weights,
                        outside_weights=outside_weights)(x, t)
