# Mofidied by:
# Copyright (c) 2017 Yuki Furuta
#
# Original work by:
# --------------------------------------------------------
# YOLOv2
# Copyright (c) 2017 leetenki
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/leetenki/YOLOv2
# --------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from chainer import function

class SumOfSquaredError(function.Function):

    """Sum of squared error (a.k.a. Euclidean loss) function."""

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel() # batch_size x input_size
        return np.array(diff.dot(diff) / 2, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return diff.dot(diff) / diff.dtype.type(2),

    def backward(self, inputs, gy):
        """
        x = inputs[0]
        t = inputs[1]
        """
        gx0 = self.diff
        return gx0, -gx0

def sum_of_squared_error(x0, x1):
    """
    Sum of squared error function.
    This function computes sum of squared error between two variables.
    """
    return SumOfSquaredError()(x0, x1)
