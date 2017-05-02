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
from chainer.links.model.vision.vgg import VGG16Layers
from chainer.links.model.vision.resnet import BuildingBlock
from chainer.links.model.vision.resnet import ResNet101Layers

from chainercv.functions.smooth_l1_loss import smooth_l1_loss

from chainercv.links.faster_rcnn.bbox_transform import bbox_transform_inv
from chainercv.links.faster_rcnn.bbox_transform import clip_boxes
from chainercv.links.faster_rcnn.nms_cpu import nms_cpu as nms
from chainercv.links.faster_rcnn.proposal_target_layer import\
    ProposalTargetLayer
from chainercv.links.faster_rcnn.region_proposal_network import\
    RegionProposalNetwork


class FasterRCNNBase(chainer.Chain):

    def __init__(
            self, feature, rpn, head,
            n_class, roi_size,
            nms_thresh=0.3,
            conf_thresh=0.05,
            sigma=1.,
            spatial_scale=0.0625,
            targets_precomputed=True
    ):
        super(FasterRCNNBase, self).__init__(
            feature=feature,
            rpn=rpn,
            head=head,
        )
        self.proposal_target_layer = ProposalTargetLayer(n_class)
        self.n_class = n_class
        self.train = True

        self.sigma = sigma
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.targets_precomputed = targets_precomputed

    def __call__(self, x, bbox=None, label=None, scale=1.):
        train = self.train and bbox is not None
        if not train:
            bbox = None

        # TODO(yuyu2172) this is really ugly
        if isinstance(scale, chainer.Variable):
            scale = np.asscalar(cuda.to_cpu(scale.data))

        # TODO(yuyu2172) this is really ugly
        if isinstance(x, chainer.Variable):
            x_data = x.data
        else:
            x_data = x
        device = cuda.get_device(x_data)

        xp = cuda.get_array_module(x)
        if bbox is not None and label is not None:
            bboxes = xp.concatenate(
                (bbox.data, label.data[:, :, None]), axis=2)
        else:
            bboxes = None
        # TODO(yuyu2172) name bboxes is really bad.

        img_size = x.shape[2:][::-1]

        h = self._extract_feature(x)

        if train:
            if isinstance(bboxes, chainer.Variable):
                bboxes = bboxes.data
            bboxes = cuda.to_cpu(bboxes)
            rpn_cls_loss, rpn_loss_bbox, rois = self.rpn(
                h, img_size, bbox=bboxes, scale=scale)
        else:
            # shape (300, 5)
            # the second axis is (batch_id, x_min, y_min, x_max, y_max)
            rois = self.rpn(h, img_size, bboxes=bboxes, scale=scale)

        if train:
            rois, labels, bbox_targets, bbox_inside_weights, \
                bbox_outside_weights = self.proposal_target_layer(
                    rois, bboxes)

        # Convert rois
        if device.id >= 0:
            rois = cuda.to_gpu(rois, device=device)

        # RCNN
        pool5 = F.roi_pooling_2d(
            h, rois, self.roi_size, self.roi_size, self.spatial_scale)
        bbox_pred, cls_score = self.head(pool5, train=train)

        if not train:
            boxes = rois[:, 1:5]
            boxes = boxes / scale
            W, H = img_size
            bbox_pred = bbox_pred.data 

            if self.targets_precomputed:
                mean = xp.tile(xp.array(self.proposal_target_layer.BBOX_NORMALIZE_MEANS), self.n_class)
                std = xp.tile(np.array(self.proposal_target_layer.BBOX_NORMALIZE_STDS), self.n_class)
                bbox_pred = (bbox_pred * std + mean).astype(np.float32)

            pred_boxes = bbox_transform_inv(boxes, bbox_pred, device.id)
            # Use this if you want to have identical results to the caffe
            # implementation.
            # pred_boxes = bbox_transform_inv(
            #     cuda.to_cpu(boxes), cuda.to_cpu(bbox_pred.data))

            cls_prob = F.softmax(cls_score)
            pred_boxes = clip_boxes(
                pred_boxes, (W / scale, H / scale), device.id)
            return pred_boxes[None], cls_prob[None].data 

        if device.id >= 0:
            labels = cuda.to_gpu(labels, device=device)
            bbox_targets = cuda.to_gpu(bbox_targets, device=device)
            bbox_inside_weights = cuda.to_gpu(
                bbox_inside_weights, device=device)
            bbox_outside_weights = cuda.to_gpu(
                bbox_outside_weights, device=device)

        loss_cls = F.softmax_cross_entropy(cls_score, labels)
        loss_bbox = smooth_l1_loss(
            bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,
            self.sigma)

        loss = rpn_cls_loss + rpn_loss_bbox + loss_bbox + loss_cls
        chainer.reporter.report({'rpn_loss_cls': rpn_cls_loss,
                                 'rpn_loss_bbox': rpn_loss_bbox,
                                 'loss_bbox': loss_bbox,
                                 'loss_cls': loss_cls,
                                 'loss': loss},
                                self)
        return loss

    def predict_bbox(self, x, scale=1.):
        """Predicts bounding boxes which satisfy confidence constraints.

        """
        pred_bbox, cls_prob = self.__call__(x, scale=scale)
        cls_prob = chainer.cuda.to_cpu(cls_prob)[0]
        pred_bbox = chainer.cuda.to_cpu(pred_bbox)[0]

        out_bbox, out_label, out_confidence = _predict_to_bbox(
            pred_bbox, cls_prob, self.nms_thresh, self.conf_thresh,
            n_class=self.n_class)
        return out_bbox[None], out_label[None], out_confidence[None]

    def _extract_feature(self, x):
        raise NotImplementedError


def _predict_to_bbox(pred_bbox, cls_prob, nms_thresh, conf_thresh, n_class):
    assert cls_prob.ndim == 2
    out_bbox = []
    out_label = []
    out_confidence = []
    # skip cls_id = 0 because it is the background class
    for cls_id in range(1, n_class):
        _cls = cls_prob[:, cls_id][:, None]  # (300, 1)
        _bbx = pred_bbox[:, cls_id * 4: (cls_id + 1) * 4]  # (300, 4)
        dets = np.hstack((_bbx, _cls))  # (300, 5)
        inds = np.where(dets[:, -1] > conf_thresh)[0]
        dets = dets[inds, :]
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        if len(dets) > 0:
            out_bbox.append(dets[:, :4])
            out_label.append(np.ones((len(dets),)) * cls_id)
            out_confidence.append(dets[:, 4])
    if len(out_bbox) != 0:
        out_bbox = np.concatenate(out_bbox, axis=0).astype(np.float32)
        out_label = np.concatenate(out_label, axis=0).astype(np.int32)
        out_confidence = np.concatenate(out_confidence, axis=0).astype(np.float32)
    else:
        out_bbox = np.zeros((0, 4), dtype=np.float32)
        out_label = np.zeros((0,), dtype=np.int32)
        out_confidence = np.zeros((0,), dtype=np.float32)
    return out_bbox, out_label, out_confidence


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
