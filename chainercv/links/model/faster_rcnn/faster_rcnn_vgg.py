import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import VGG16Layers

from chainercv.links.model.faster_rcnn.faster_rcnn import FasterRCNNBase
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork


class FasterRCNNVGG16(FasterRCNNBase):

    """FasterRCNN based on VGG16.

    """

    feat_stride = 16

    def __init__(self,
                 n_class,
                 pretrained_model='imagenet',
                 nms_thresh=0.3, score_thresh=0.7,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 proposal_creator_params={}
                 ):
        if pretrained_model:
            init = chainer.initializers.constant.Zero()
            vgg_kwargs = {'initialW': init, 'initial_bias': init}
        else:
            vgg_kwargs = {}
        bbox_kwargs = {'initialW': chainer.initializers.Normal(0.001)}
        score_kwargs = {'initialW': chainer.initializers.Normal(0.01)}

        feature = VGG16FeatureExtractor(extract='conv5_3')
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            proposal_creator_params=proposal_creator_params,
        )
        head = VGG16RoIPoolingHead(
            n_class,
            roi_size=7, spatial_scale=1. / self.feat_stride,
            vgg_kwargs=vgg_kwargs,
            bbox_kwargs=bbox_kwargs,
            score_kwargs=score_kwargs
        )

        super(FasterRCNNVGG16, self).__init__(
            feature,
            rpn,
            head,
            n_class=n_class,
            mean=np.array([102.9801, 115.9465, 122.7717],
                          dtype=np.float32)[:, None, None],
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
        )

        if pretrained_model == 'imagenet':
            self._copy_imagenet_pretrained_vgg16()

    def _copy_imagenet_pretrained_vgg16(self):
        pretrained_model = VGG16Layers()
        self.feature.conv1_1.copyparams(pretrained_model.conv1_1)
        self.feature.conv1_2.copyparams(pretrained_model.conv1_2)
        self.feature.conv2_1.copyparams(pretrained_model.conv2_1)
        self.feature.conv2_2.copyparams(pretrained_model.conv2_2)
        self.feature.conv3_1.copyparams(pretrained_model.conv3_1)
        self.feature.conv3_2.copyparams(pretrained_model.conv3_2)
        self.feature.conv3_3.copyparams(pretrained_model.conv3_3)
        self.feature.conv4_1.copyparams(pretrained_model.conv4_1)
        self.feature.conv4_2.copyparams(pretrained_model.conv4_2)
        self.feature.conv4_3.copyparams(pretrained_model.conv4_3)
        self.feature.conv5_1.copyparams(pretrained_model.conv5_1)
        self.feature.conv5_2.copyparams(pretrained_model.conv5_2)
        self.feature.conv5_3.copyparams(pretrained_model.conv5_3)
        self.head.fc6.copyparams(pretrained_model.fc6)
        self.head.fc7.copyparams(pretrained_model.fc7)


class VGG16RoIPoolingHead(chainer.Chain):

    """Regress and classify bounding boxes based on RoI pooled features.

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 vgg_kwargs, bbox_kwargs, score_kwargs):
        super(VGG16RoIPoolingHead, self).__init__(
            fc6=L.Linear(25088, 4096, **vgg_kwargs),
            fc7=L.Linear(4096, 4096, **vgg_kwargs),
            bbox=L.Linear(4096, n_class * 4, **bbox_kwargs),
            score=L.Linear(4096, n_class, **score_kwargs),
        )
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois, batch_indices, train=False):
        """Pool and forward batches of patches.

        Args:
            x (~chainer.Variable):
            rois (array)
            batch_indices (array)

        Returns:
            list of chainer.Variable, list of chainer.Variable

        """
        batch_indices = batch_indices.astype(np.float32)
        rois = self.xp.concatenate(
            (batch_indices[:, None], rois), axis=1)
        pool = F.roi_pooling_2d(
            x, rois, self.roi_size, self.roi_size, self.spatial_scale)

        fc6 = F.dropout(F.relu(self.fc6(pool)), train=train)
        fc7 = F.dropout(F.relu(self.fc7(fc6)), train=train)
        roi_bboxes = self.bbox(fc7)
        roi_scores = self.score(fc7)

        return roi_bboxes, roi_scores


class VGG16FeatureExtractor(VGG16Layers):

    """Truncated VGG that extracts an intermediate feature.

    """

    def __init__(self, extract='conv5_3', vgg_kwargs={}):
        super(VGG16FeatureExtractor, self).__init__(pretrained_model=False)
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
