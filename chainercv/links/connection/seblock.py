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

    def forward(self, u):
        B, C, H, W = u.shape

        z = F.average(u, axis=(2, 3))
        x = F.relu(self.down(z))
        x = F.sigmoid(self.up(x))
        x = F.reshape(x, x.shape[:2] + (1, 1))
        # Spatial axes of `x` will be broadcasted.
        return u * x
