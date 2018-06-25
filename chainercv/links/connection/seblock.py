import chainer
import chainer.functions as F
import chainer.links as L


class SEBlock(chainer.Chain):

    """A squeeze-and-excitation block.

    This block is part of squeeze-and-excitation networks. Channel-wise
    multiplication weights are inferred from and applied to input feature map.
    Please refer to `the original paper
    <https://arxiv.org/pdf/1709.01507.pdf>`_ for more details.

    .. seealso::
        :class:`chainercv.links.model.senet.SEResNet`

    Args:
        n_channel (int): The number of channels of the input and output array.
        ratio (int): Reduction ratio of :obj:`n_channel` to the number of
            hidden layer units.

    """

    def __init__(self, n_channel, ratio=16):

        super(SEBlock, self).__init__()
        reduction_size = n_channel // ratio

        with self.init_scope():
            self.down = L.Linear(n_channel, reduction_size)
            self.up = L.Linear(reduction_size, n_channel)

    def __call__(self, u):
        B, C, H, W = u.shape

        z = _global_average_pooling_2d(u)
        x = F.relu(self.down(z))
        x = F.sigmoid(self.up(x))

        x = F.broadcast_to(x, (H, W, B, C))
        x = x.transpose((2, 3, 0, 1))

        return u * x


def _global_average_pooling_2d(x):
    B, C, H, W = x.data.shape
    h = F.average_pooling_2d(x, (H, W), stride=1)
    h = h.reshape((B, C))
    return h
