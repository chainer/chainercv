from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links.model.fpn.faster_rcnn import FasterRCNN
from chainercv.links.model.fpn.fpn import FPN
from chainercv.links.model.fpn.bbox_head import BboxHead
from chainercv.links.model.fpn.rpn import RPN
from chainercv.links.model.resnet import ResNet101
from chainercv.links.model.resnet import ResNet50
from chainercv import utils


class FasterRCNNFPNResNet(FasterRCNN):
    """Base class for FasterRCNNFPNResNet50 and FasterRCNNFPNResNet101.

    A subclass of this class should have :obj:`_base` and :obj:`_models`.
    """

    def __init__(self, n_fg_class=None, pretrained_model=None,
                 min_size=800, max_size=1333):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        base = self._base(n_class=1, arch='he')
        base.pick = ('res2', 'res3', 'res4', 'res5')
        base.pool1 = lambda x: F.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False)
        base.remove_unused()
        extractor = FPN(
            base, len(base.pick), (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64))

        super(FasterRCNNFPNResNet, self).__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            bbox_head=BboxHead(param['n_fg_class'] + 1, extractor.scales),
            min_size=min_size, max_size=max_size
        )

        if path == 'imagenet':
            _copyparams(
                self.extractor.base,
                self._base(pretrained_model='imagenet', arch='he'))
        elif path:
            chainer.serializers.load_npz(path, self)


class FasterRCNNFPNResNet50(FasterRCNNFPNResNet):
    """Feature Pyramid Networks with ResNet-50.

    This is a model of Feature Pyramid Networks [#]_.
    This model uses :class:`~chainercv.links.ResNet50` as
    its base feature extractor.

    .. [#] Tsung-Yi Lin et al.
       Feature Pyramid Networks for Object Detection. CVPR 2017

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
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please \
            refer to :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.

    """

    _base = ResNet50
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': 'https://chainercv-models.preferred.jp/'
            'faster_rcnn_fpn_resnet50_coco_trained_2018_12_13.npz',
            'cv2': True
        },
    }


class FasterRCNNFPNResNet101(FasterRCNNFPNResNet):
    """Feature Pyramid Networks with ResNet-101.

    This is a model of Feature Pyramid Networks [#]_.
    This model uses :class:`~chainercv.links.ResNet101` as
    its base feature extractor.

    .. [#] Tsung-Yi Lin et al.
       Feature Pyramid Networks for Object Detection. CVPR 2017

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
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please \
            refer to :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.

    """

    _base = ResNet101
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': 'https://chainercv-models.preferred.jp/'
            'faster_rcnn_fpn_resnet101_coco_trained_2018_12_13.npz',
            'cv2': True
        },
    }


def _copyparams(dst, src):
    if isinstance(dst, chainer.Chain):
        for link in dst.children():
            _copyparams(link, src[link.name])
    elif isinstance(dst, chainer.ChainList):
        for i, link in enumerate(dst):
            _copyparams(link, src[i])
    else:
        dst.copyparams(src)
        if isinstance(dst, L.BatchNormalization):
            dst.avg_mean = src.avg_mean
            dst.avg_var = src.avg_var
