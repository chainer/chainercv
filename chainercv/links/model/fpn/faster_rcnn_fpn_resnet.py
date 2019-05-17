from __future__ import division

import copy

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links.model.fpn.bbox_head import BboxHead
from chainercv.links.model.fpn.faster_rcnn import FasterRCNN
from chainercv.links.model.fpn.fpn import FPN
from chainercv.links.model.fpn.mask_head import MaskHead
from chainercv.links.model.fpn.rpn import RPN
from chainercv.links.model.resnet import ResNet101
from chainercv.links.model.resnet import ResNet50
from chainercv import utils


class FasterRCNNFPNResNet(FasterRCNN):
    """Base class for Faster R-CNN with a ResNet backbone and FPN.

    A subclass of this class should have :obj:`_base` and :obj:`_models`.

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
        return_values (list of strings): Determines the values
            returned by :meth:`predict`.
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please \
            refer to :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.

    """

    def __init__(self, n_fg_class=None, pretrained_model=None,
                 return_values=['bboxes', 'labels', 'scores'],
                 min_size=800, max_size=1333):
        param, path = utils.prepare_model_param(locals(), self._models)

        base_param = copy.deepcopy(self._base.preset_params['imagenet'])
        base_param['n_class'] = 1
        base = self._base(arch='he', **base_param)
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
            mask_head=MaskHead(param['n_fg_class'] + 1, extractor.scales),
            return_values=return_values,
            min_size=min_size, max_size=max_size
        )

        if path == 'imagenet':
            _copyparams(
                self.extractor.base,
                self._base(pretrained_model='imagenet', arch='he'))
        elif path:
            chainer.serializers.load_npz(path, self)


class MaskRCNNFPNResNet(FasterRCNNFPNResNet):
    """Mask R-CNN with a ResNet backbone and FPN.

    Please refer to :class:`~chainercv.links.model.fpn.FasterRCNNFPNResNet`.

    """

    def __init__(self, n_fg_class=None, pretrained_model=None,
                 return_values=['masks', 'labels', 'scores'],
                 min_size=800, max_size=1333):
        super(MaskRCNNFPNResNet, self).__init__(
            n_fg_class, pretrained_model, return_values,
            min_size, max_size)


class FasterRCNNFPNResNet50(FasterRCNNFPNResNet):
    """Faster R-CNN with ResNet-50 and FPN.

    Please refer to :class:`~chainercv.links.model.fpn.FasterRCNNFPNResNet`.

    """

    preset_params = {
        'coco': {'n_fg_class': 80},
    }
    _base = ResNet50
    _models = {
        'coco': {
            'param': preset_params['coco'],
            'url': 'https://chainercv-models.preferred.jp/'
            'faster_rcnn_fpn_resnet50_coco_trained_2019_03_15.npz',
            'cv2': True
        },
    }


class FasterRCNNFPNResNet101(FasterRCNNFPNResNet):
    """Faster R-CNN with ResNet-101 and FPN.

    Please refer to :class:`~chainercv.links.model.fpn.FasterRCNNFPNResNet`.

    """

    preset_params = {
        'coco': {'n_fg_class': 80},
    }
    _base = ResNet101
    _models = {
        'coco': {
            'param': preset_params['coco'],
            'url': 'https://chainercv-models.preferred.jp/'
            'faster_rcnn_fpn_resnet101_coco_trained_2019_03_15.npz',
            'cv2': True
        },
    }


class MaskRCNNFPNResNet50(MaskRCNNFPNResNet):
    """Mask R-CNN with ResNet-50 and FPN.

    Please refer to :class:`~chainercv.links.model.fpn.FasterRCNNFPNResNet`.

    """

    preset_params = {
        'coco': {'n_fg_class': 80},
    }
    _base = ResNet50
    _models = {
        'coco': {
            'param': preset_params['coco'],
            'url': 'https://chainercv-models.preferred.jp/'
            'mask_rcnn_fpn_resnet50_coco_trained_2019_03_15.npz',
            'cv2': True
        },
    }


class MaskRCNNFPNResNet101(MaskRCNNFPNResNet):
    """Mask R-CNN with ResNet-101 and FPN.

    Please refer to :class:`~chainercv.links.model.fpn.FasterRCNNFPNResNet`.

    """

    preset_params = {
        'coco': {'n_fg_class': 80},
    }
    _base = ResNet101
    _models = {
        'coco': {
            'param': preset_params['coco'],
            'url': 'https://chainercv-models.preferred.jp/'
            'mask_rcnn_fpn_resnet101_coco_trained_2019_03_15.npz',
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
