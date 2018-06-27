import chainer.functions as F


def global_average_pooling_2d(x):
    """Two-dimensional global average pooling function.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (N: batch, C: channel, H, height, W: width).

    """

    B, C, H, W = x.data.shape
    h = F.average_pooling_2d(x, (H, W), stride=1)
    h = h.reshape((B, C))
    return h
