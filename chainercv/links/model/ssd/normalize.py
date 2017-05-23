import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers


class Normalize(chainer.Link):
    """Learnable L2 normalization [#]_.

    This link normalizes input along the channel axis and scales it.
    The scale factors are trained channel-wise.

    .. [#] Wei Liu, Andrew Rabinovich, Alexander C. Berg.
       ParseNet: Looking Wider to See Better. ICLR 2016.

    Args:
        n_channel (int): The number of channels.
        initial: A value to initialize the scale factors. It is pased to
            :meth:`chainer.initializers._get_initializer`. The default value
            is 0.
        eps (float): A small value to avoid zero-division. The default value
            is :math:`1e-5`.

    """

    def __init__(self, n_channel, initial=0, eps=1e-5):
        super(Normalize, self).__init__()
        self.eps = eps
        self.add_param(
            'scale', n_channel,
            initializer=initializers._get_initializer(initial))

    def __call__(self, x):
        """Normalize input and scale it.

        Args:
            x (chainer.Variable): A variable holding 4-dimensional array.
                Its :obj:`dtype` is :obj:`numpy.float32`.

        Returns:
            chainer.Variable:
            The shape and :obj:`dtype` are same as those of input.
        """

        x = F.normalize(x, eps=self.eps, axis=1)
        scale = F.broadcast_to(self.scale[:, np.newaxis, np.newaxis], x.shape)
        return x * scale
