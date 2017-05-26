import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Multibox(chainer.Chain):
    """Multibox head of Single Shot Multibox Detector.

    This is a head part of Single Shot Multibox Detector [#]_.
    This link computes :obj:`loc` and :obj:`conf` from feature maps.
    :obj:`loc` contains information of the coordinates of bounding boxes
    and :obj:`conf` contains that of classes.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.GlorotUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(Multibox, self).__init__(
            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )

        if initialW is None:
            initialW = initializers.GlorotUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(None, n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                None, n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`loc` and :obj:`conf`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`, :obj:`loc` and
            :obj:`conf`. :obj:`loc` is an array whose shape is :math:`(R, 4)`,
            where :math:`R` is the number of default bounding boxes.
            :obj:`conf` is an array whose shape is :math:`(R, n\_class)`
        """

        ys_loc = list()
        ys_conf = list()
        for i, x in enumerate(xs):
            loc = self.loc[i](x)
            loc = F.transpose(loc, (0, 2, 3, 1))
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            ys_loc.append(loc)

            conf = self.conf[i](x)
            conf = F.transpose(conf, (0, 2, 3, 1))
            conf = F.reshape(
                conf, (conf.shape[0], -1, self.n_class))
            ys_conf.append(conf)

        y_loc = F.concat(ys_loc, axis=1)
        y_conf = F.concat(ys_conf, axis=1)

        return y_loc, y_conf
