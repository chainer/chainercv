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
        n_channels (int): The number of channels of the input and output array.
        ratio (int): Reduction ratio of n_channels to the number of hidden
            layer units.

    """

    def __init__(self, n_channels, ratio=16):

        super(SEBlock, self).__init__()
        reduction_size = n_channels // ratio

        with self.init_scope():
            self.down = L.Linear(n_channels, reduction_size)
            self.up = L.Linear(reduction_size, n_channels)

    def __call__(self, u):
        n_batch, n_channels, height, width = u.shape

        z = F.average(u, axis=(2, 3))
        x = F.relu(self.down(z))
        x = F.sigmoid(self.up(x))

        x = F.broadcast_to(x, (height, width, n_batch, n_channels))
        x = x.transpose((2, 3, 0, 1))

        return u * x
