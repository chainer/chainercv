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

    _base = ResNet50
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': None,
            'cv2': True
        },
    }


class MaskRCNNFPNResNet101(MaskRCNNFPNResNet):

    _base = ResNet101
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': None,
            'cv2': True
        },
    }
