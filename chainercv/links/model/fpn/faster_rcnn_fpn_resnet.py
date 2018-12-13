import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links.model.fpn.head import Head
from chainercv.links.model.fpn.faster_rcnn import FasterRCNN
from chainercv.links.model.fpn.fpn import FPN
from chainercv.links.model.fpn.rpn import RPN
from chainercv.links.model.resnet import ResNet101
from chainercv.links.model.resnet import ResNet50
from chainercv import utils


class FasterRCNNFPNResNet(FasterRCNN):

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

        super().__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(param['n_fg_class'] + 1, extractor.scales),
        )

        if path == 'imagenet':
            _copyparams(
                self.extractor.base,
                self._base(pretrained_model='imagenet', arch='he'))
        elif path:
            chainer.serializers.load_npz(path, self)


class FasterRCNNFPNResNet50(FasterRCNNFPNResNet):

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
