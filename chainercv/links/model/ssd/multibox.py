import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Multibox(chainer.Chain):
    """Multibox head of Single Shot Multibox Detector.

    This is a head part of Single Shot Multibox Detector [#]_.
    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(Multibox, self).__init__()
        with self.init_scope():
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def forward(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = []
        mb_confs = []
        for i, x in enumerate(xs):
            mb_loc = self.loc[i](x)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](x)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs
