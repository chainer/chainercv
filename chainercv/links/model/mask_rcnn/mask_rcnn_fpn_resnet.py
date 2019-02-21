from __future__ import division

import chainer
import chainer.functions as F

from chainercv.links.model.fpn import FPN
from chainercv.links.model.fpn import Head
from chainercv.links.model.fpn import RPN
from chainercv.links.model.mask_rcnn.mask_head import MaskHead
from chainercv.links.model.mask_rcnn.mask_rcnn import MaskRCNN
from chainercv.links.model.resnet import ResNet101
from chainercv.links.model.resnet import ResNet50
from chainercv import utils

from chainercv.links.model.fpn.faster_rcnn_fpn_resnet import _copyparams


class MaskRCNNFPNResNet(MaskRCNN):

    """Base class for Mask R-CNN with ResNet backbone.

    A subclass of this class should have :obj:`_base` and :obj:`_models`.
    """

    def __init__(self, n_fg_class=None, pretrained_model=None):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        base = self._base(n_class=1, arch='he')
        base.pick = ('res2', 'res3', 'res4', 'res5')
        base.pool1 = lambda x: F.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False)
        base.remove_unused()
        extractor = FPN(
            base, len(base.pick), (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64))

        n_class = param['n_fg_class'] + 1
        super(MaskRCNNFPNResNet, self).__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_class, extractor.scales),
            mask_head=MaskHead(n_class, extractor.scales)
        )
        if path == 'imagenet':
            _copyparams(
                self.extractor.base,
                self._base(pretrained_model='imagenet', arch='he'))
        elif path:
            chainer.serializers.load_npz(path, self)


class MaskRCNNFPNResNet50(MaskRCNNFPNResNet):

    """Mask R-CNN with ResNet-50.

    This is a model of Mask R-CNN [#]_.
    This model uses :class:`~chainercv.links.ResNet50` as
    its base feature extractor.

    .. [#] Kaiming He et al. Mask R-CNN. ICCV 2017

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (string): The weight file to be loaded.
           This can take :obj:`'coco'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'coco'`: Load weights trained on train split of \
                MS COCO 2017. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`80` or :obj:`None`.
            * :obj:`'imagenet'`: Load weights of ResNet-50 trained on \
                ImageNet. \
                The weight file is downloaded and cached automatically. \
                This option initializes weights partially and the rests are \
                initialized randomly. In this case, :obj:`n_fg_class` \
                can be set to any number.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _base = ResNet50
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': None,
            'cv2': True
        },
    }


class MaskRCNNFPNResNet101(MaskRCNNFPNResNet):

    """Mask R-CNN with ResNet-101.

    This is a model of Mask R-CNN [#]_.
    This model uses :class:`~chainercv.links.ResNet101` as
    its base feature extractor.

    .. [#] Kaiming He et al. Mask R-CNN. ICCV 2017

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (string): The weight file to be loaded.
           This can take :obj:`'coco'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'coco'`: Load weights trained on train split of \
                MS COCO 2017. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`80` or :obj:`None`.
            * :obj:`'imagenet'`: Load weights of ResNet-101 trained on \
                ImageNet. \
                The weight file is downloaded and cached automatically. \
                This option initializes weights partially and the rests are \
                initialized randomly. In this case, :obj:`n_fg_class` \
                can be set to any number.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _base = ResNet101
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': None,
            'cv2': True
        },
    }
