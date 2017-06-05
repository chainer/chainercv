import chainer

from chainercv.links.model.ssd import multibox_loss


class MultiboxTrainChain(chainer.Chain):
    """Calculate multibox lossesand report them.

    This is used to train Single Shot Multibox Detector in the joint training
    scheme [#]_.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        model (~chainercv.models.link.ssd.SSD): An instance of
            :class:`~chainercv.models.link.ssd.SSD`.
        alpha (float): A float which determines the balance between
            localization and classification. The loss is defined as
            :obj:`alpha * loc_loss + conf_loss`. The default value
            is :obj:`1`. This is the value used in the original paper.
        k (float): A float which is used by
            :func:`~chainercv.links.model.ssd.multibox_loss`.
            The default value is :obj:`3`.
            This is the value used in the original paper.
    """

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__(model=model)
        self.alpha = alpha
        self.k = k

    def __call__(self, x, gt_mb_locs, gt_mb_labels):
        """Forward SSD and calculate losses.

        Args:
             x (~chainer.Variable): A variable holding a batch of images.
                 The sizes of all images should be :obj:`model.insize`.
             gt_mb_locs (~chainer.Variable): A variable holding the locations
                 of ground truth.
             gt_mb_labels (~chainer.Variable): A variable holding the classes
                 of ground truth.

        Returns:
            chainer.Variable:
            A scalar variable holding the loss value.
        """

        mb_locs, mb_confs = self.model(x)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loc_loss': loc_loss, 'conf_loss': conf_loss},
            self)

        return loss
