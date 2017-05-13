import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import VGG16Layers

from chainercv.links.model.faster_rcnn.faster_rcnn import FasterRCNNBase
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork


class FasterRCNNVGG16(FasterRCNNBase):

    feat_stride = 16

    def __init__(self,
                 n_class,
                 pretrained_model='auto',
                 nms_thresh=0.3, score_thresh=0.7,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 proposal_creator_params={}
                 ):
        feature = VGG16FeatureExtractor(extract='conv5_3')
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            proposal_creator_params=proposal_creator_params,
        )
        head = FasterRCNNHeadVGG16(
            n_class,
            bbox_initialW=chainer.initializers.Normal(0.001),
            cls_initialW=chainer.initializers.Normal(0.01),
        )

        super(FasterRCNNVGG16, self).__init__(
            feature,
            rpn,
            head,
            n_class=n_class,
            roi_size=7,
            spatial_scale=1. / self.feat_stride,
            mean=np.array([[[102.9801, 115.9465, 122.7717]]]),
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
        )


class FasterRCNNHeadVGG16(chainer.Chain):

    """Regress and classify bounding boxes based on RoI pooled features.

    """

    def __init__(self, n_class,
                 fc_initialW=None, cls_initialW=None, bbox_initialW=None):

        super(FasterRCNNHeadVGG16, self).__init__(
            # these linear links take some time to initialize
            fc6=L.Linear(25088, 4096, initialW=fc_initialW),
            fc7=L.Linear(4096, 4096, initialW=fc_initialW),
            bbox_tf=L.Linear(4096, n_class * 4, initialW=bbox_initialW),
            cls_score=L.Linear(4096, n_class, initialW=cls_initialW),
        )

    def __call__(self, x, train=False):
        fc6 = F.dropout(F.relu(self.fc6(x)), train=train)
        fc7 = F.dropout(F.relu(self.fc7(fc6)), train=train)

        bbox_tf = self.bbox_tf(fc7)
        score = self.cls_score(fc7)
        return bbox_tf, score


class VGG16FeatureExtractor(VGG16Layers):

    """Truncated VGG that extracts an intermediate feature.

    """

    def __init__(self, extract='conv5_3', pretrained_model=False):
        super(VGG16FeatureExtractor, self).__init__(pretrained_model)
        self.extract = extract
        if self.extract not in self.functions:
            raise ValueError(
                'Only the features produced by VGG can be extracted')

        start_deleting = False
        delete_layers = []
        for layer in self.functions.keys():
            if start_deleting:
                delete_layers.append(layer)
            if layer == extract:
                start_deleting = True

        for layer in delete_layers:
            self.functions.pop(layer)
            if layer in self._children:
                self._children.remove(layer)
                delattr(self, layer)

    def __call__(self, x, train=False):
        hs = super(VGG16FeatureExtractor, self).__call__(
            x, layers=[self.extract], test=not train)
        return hs[self.extract]
