# Original work by:
# ----------------------------------------------------------------------------
# Copyright (c) 2016 Shunta Saito
# Licensed under The MIT License
# https://github.com/mitmul/chainer-faster-rcnn
# ----------------------------------------------------------------------------

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.initializers import constant
import chainer.links as L
from chainer.links.model.vision.resnet import BuildingBlock
from chainer.links.model.vision.resnet import ResNet101Layers
from chainer.links.model.vision.vgg import VGG16Layers

from chainercv.links.faster_rcnn.bbox_transform import bbox_transform_inv
from chainercv.links.faster_rcnn.bbox_transform import clip_boxes
from chainercv.links.faster_rcnn.nms_cpu import nms_cpu as nms

from chainercv.links.faster_rcnn.region_proposal_network import\
    RegionProposalNetwork


class FasterRCNNBase(chainer.Chain):

    def __init__(
            self, feature, rpn, head,
            n_class, roi_size,
            nms_thresh=0.3,
            conf_thresh=0.05,
            spatial_scale=0.0625,
            targets_precomputed=True,
            bbox_normalize_mean=(0., 0., 0., 0.),
            bbox_normalize_std=(0.1, 0.1, 0.2, 0.2),
            proposal_target_layer_params={},
    ):
        super(FasterRCNNBase, self).__init__(
            feature=feature,
            rpn=rpn,
            head=head,
        )
        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.targets_precomputed = targets_precomputed
        self.bbox_normalize_mean = bbox_normalize_mean
        self.bbox_normalize_std = bbox_normalize_std

    def __call__(self, x, scale=1., layers=['bbox', 'cls_prob'],
                 rpn_only=False, test=True):
        # Making stopabble __call__ like that of VGG is not practical
        # in the case of FasterRCNN.
        # This is because there are few number of cases where you
        # need to extract intermediate representations.
        # I can only come up with two cases.
        # 1. Stop at RPN
        # 2. Forward all

        # These simple control logic can be expressed using a simple flag
        # variable.
        # Therefore, an orderedDict used in VGG is not necessary.

        # TODO(yuyu2172)  stop using scale filtering by default call.
        # TODO(yuyu2172)  support a scenario where RoI is already computed.
        activations = {key: None for key in layers}

        if isinstance(scale, chainer.Variable):
            scale = scale.data
        scale = np.asscalar(cuda.to_cpu(np.array(scale)))
        img_size = x.shape[2:][::-1]

        h = self._extract_feature(x)
        rpn_bbox_pred, rpn_cls_score, roi, anchor =\
            self.rpn(h, img_size, scale, train=not test)

        _update_if_specified({
            'feature': h,
            'rpn_bbox_pred': rpn_bbox_pred,
            'rpn_cls_score': rpn_cls_score,
            'roi': roi,
            'anchor': anchor
        }, activations)
        if rpn_only:
            return activations

        pool5 = F.roi_pooling_2d(
            h, roi, self.roi_size, self.roi_size, self.spatial_scale)
        bbox_tf, cls_score = self.head(pool5, train=False)

        xp = chainer.cuda.get_array_module(pool5)
        device = chainer.cuda.get_device(pool5.data)
        # Convert predictions to bounding boxes in image coordinates.
        bbox_roi = roi[:, 1:5]
        bbox_roi = bbox_roi / scale
        bbox_tf_data = bbox_tf.data
        if self.targets_precomputed:
            mean = xp.tile(
                xp.array(self.bbox_normalize_mean),
                self.n_class)
            std = xp.tile(
                np.array(self.bbox_normalize_std),
                self.n_class)
            bbox_tf_data = (bbox_tf_data * std + mean).astype(np.float32)
        bbox = bbox_transform_inv(bbox_roi, bbox_tf_data, device.id)
        W, H = img_size
        bbox = clip_boxes(
            bbox, (W / scale, H / scale), device.id)

        # Compute probabilities that each bounding box is assigned to.
        cls_prob = F.softmax(cls_score).data

        _update_if_specified({
            'pool5': pool5,
            'bbox_tf': bbox_tf,
            'cls_score': cls_score,
            'bbox': bbox[None],
            'cls_prob': cls_prob[None]
        }, activations)
        return activations

    def predict(self, x, scale=1.):
        """Predicts bounding boxes which satisfy confidence constraints.

        """
        out = self.__call__(
            x, scale=scale, layers=['bbox', 'cls_prob'])
        bbox = chainer.cuda.to_cpu(out['bbox'])[0]
        cls_prob = chainer.cuda.to_cpu(out['cls_prob'])[0]

        assert cls_prob.ndim == 2
        bbox_nms = []
        label_nms = []
        conf_nms = []
        # skip cls_id = 0 because it is the background class
        for cls_id in range(1, self.n_class):
            _cls = cls_prob[:, cls_id][:, None]  # (300, 1)
            _bbx = bbox[:, cls_id * 4: (cls_id + 1) * 4]  # (300, 4)
            dets = np.hstack((_bbx, _cls))  # (300, 5)
            inds = np.where(dets[:, -1] > self.conf_thresh)[0]
            dets = dets[inds, :]
            keep = nms(dets, self.nms_thresh)
            dets = dets[keep, :]
            if len(dets) > 0:
                bbox_nms.append(dets[:, :4])
                label_nms.append(np.ones((len(dets),)) * cls_id)
                conf_nms.append(dets[:, 4])
        if len(bbox_nms) != 0:
            bbox_nms = np.concatenate(bbox_nms, axis=0).astype(np.float32)
            label_nms = np.concatenate(label_nms, axis=0).astype(np.int32)
            conf_nms = np.concatenate(conf_nms, axis=0).astype(np.float32)
        else:
            bbox_nms = np.zeros((0, 4), dtype=np.float32)
            label_nms = np.zeros((0,), dtype=np.int32)
            conf_nms = np.zeros((0,), dtype=np.float32)

        return bbox_nms[None], label_nms[None], conf_nms[None]

    def _extract_feature(self, x):
        raise NotImplementedError


def _update_if_specified(source, target):
    for key in source.keys():
        if key in target:
            target[key] = source[key]


class FasterRCNNHeadVGG(chainer.Chain):

    def __init__(self, n_class, initialW=None):
        cls_init = chainer.initializers.Normal(0.01)
        bbox_init = chainer.initializers.Normal(0.001)
        super(FasterRCNNHeadVGG, self).__init__(
            # these linear links take some time to initialize
            fc6=L.Linear(25088, 4096, initialW=initialW),
            fc7=L.Linear(4096, 4096, initialW=initialW),
            cls_score=L.Linear(4096, n_class, initialW=cls_init),
            bbox_pred=L.Linear(4096, n_class * 4, initialW=bbox_init)
        )

    def __call__(self, x, train=False):
        fc6 = F.dropout(F.relu(self.fc6(x)), train=train)
        fc7 = F.dropout(F.relu(self.fc7(fc6)), train=train)

        # Per class probability
        cls_score = self.cls_score(fc7)

        bbox_pred = self.bbox_pred(fc7)
        return bbox_pred, cls_score


class FasterRCNNVGG(FasterRCNNBase):

    def __init__(self, n_class=21,
                 nms_thresh=0.3, conf_thresh=0.05,
                 n_anchors=9, anchor_scales=[8, 16, 32],
                 targets_precomputed=True
                 ):
        feat_stride = 16
        rpn_sigma = 3.
        sigma = 1.

        feature = VGG16Layers()
        rpn = RegionProposalNetwork(
            512, 512, anchor_scales=anchor_scales,
            feat_stride=feat_stride,
            rpn_sigma=rpn_sigma)
        head = FasterRCNNHeadVGG(n_class, initialW=constant.Zero())
        super(FasterRCNNVGG, self).__init__(
            feature,
            rpn,
            head,
            n_class=n_class,
            roi_size=7,
            nms_thresh=nms_thresh,
            conf_thresh=conf_thresh,
            sigma=sigma,
            targets_precomputed=targets_precomputed
        )
        # Handle pretrained models
        self.head.fc6.copyparams(self.feature.fc6)
        self.head.fc7.copyparams(self.feature.fc7)

        remove_links = ['fc6', 'fc7', 'fc8']
        for name in remove_links:
            self.feature._children.remove(name)
            delattr(self.feature, name)

    def _extract_feature(self, x):
        hs = self.feature(x, layers=['pool2', 'conv5_3'])
        h = hs['conv5_3']
        hs['pool2'].unchain_backward()
        return h


class FasterRCNNHeadResNet(chainer.Chain):

    # MEMO: http://ethereon.github.io/netscope/#/gist/4b53f4ee831891ce886a5b8fae62473c
    def __init__(self, n_class, initialW=None):
        cls_init = chainer.initializers.Normal(0.01)
        bbox_init = chainer.initializers.Normal(0.001)
        super(FasterRCNNHeadResNet, self).__init__(
            res5=BuildingBlock(3, 1024, 512, 2048, 2, initialW=initialW),
            cls_score=L.Linear(2048, n_class, initialW=cls_init),
            bbox_pred=L.Linear(2048, n_class * 4, initialW=bbox_init),
        )

    def __call__(self, x, train=False):
        h = self.res5(x, test=not train)
        h = F.max_pooling_2d(h, ksize=7)

        cls_score = self.cls_score(h)
        bbox_pred = self.bbox_pred(h)
        return bbox_pred, cls_score


class FasterRCNNResNet(FasterRCNNBase):

    def __init__(self, n_class=21,
                 nms_thresh=0.3, conf_thresh=0.05,
                 n_anchors=9, anchor_scales=[8, 16, 32],
                 targets_precomputed=True
                 ):
        feat_stride = 16
        rpn_sigma = 3.
        sigma = 1.

        feature = ResNet101Layers()
        rpn = RegionProposalNetwork(
            1024, 256, feat_stride=feat_stride,
            anchor_scales=anchor_scales, rpn_sigma=rpn_sigma)
        head = FasterRCNNHeadResNet(n_class, initialW=constant.Zero())

        super(FasterRCNNResNet, self).__init__(
            feature,
            rpn,
            head,
            n_class=n_class,
            roi_size=14,
            nms_thresh=nms_thresh,
            conf_thresh=conf_thresh,
            sigma=sigma,
        )
        # Handle pretrained models
        self.head.res5.copyparams(self.feature.res5)

        remove_links = ['res5']
        for name in remove_links:
            self.feature._children.remove(name)
            delattr(self.feature, name)

    def _extract_feature(self, x):
        hs = self.feature(x, layers=['res2', 'res4'])
        h = hs['res4']
        hs['res2'].unchain_backward()
        return h
