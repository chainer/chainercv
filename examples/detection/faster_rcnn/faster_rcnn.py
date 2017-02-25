# Mofidied by:
# Copyright (c) 2017 Yusuke Niitani

# Original work by:
# ----------------------------------------------------------------------------
# Copyright (c) 2016 Shunta Saito
# Licensed under The MIT License
# https://github.com/mitmul/chainer-faster-rcnn
# ----------------------------------------------------------------------------

# from lib.faster_rcnn.bbox_transform import bbox_transform_inv
# from lib.faster_rcnn.bbox_transform import clip_boxes
# from lib.faster_rcnn.proposal_target_layer import ProposalTargetLayer
# from lib.faster_rcnn.roi_pooling_2d import roi_pooling_2d
# from lib.faster_rcnn.smooth_l1_loss import smooth_l1_loss
# from lib.models.rpn import RPN
# from lib.models.vgg16 import VGG16

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.vgg import VGG16Layers

from lib.region_proporsal.rpn import RPN
from lib.functions.roi_pooling_2d import roi_pooling_2d
from lib.functions.smooth_l1_loss import smooth_l1_loss
from lib.region_proporsal.bbox_transform import bbox_transform_inv
from lib.region_proporsal.bbox_transform import clip_boxes
from lib.region_proporsal.proposal_target_layer import ProposalTargetLayer


class FasterRCNN(chainer.Chain):

    def __init__(
            self, gpu=-1, rpn_in_ch=512, rpn_out_ch=512,
            n_anchors=9, feat_stride=16, anchor_scales=[8, 16, 32],
            num_classes=21, spatial_scale=0.0625, rpn_sigma=1.0, sigma=3.0):
        # names of links are consistent with the original implementation so
        # that learned parameters can be used
        super(FasterRCNN, self).__init__(
            trunk=VGG16Layers(),
            RPN=RPN(rpn_in_ch, rpn_out_ch, n_anchors, feat_stride,
                    anchor_scales, num_classes, rpn_sigma),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            cls_score=L.Linear(4096, num_classes),
            bbox_pred=L.Linear(4096, num_classes * 4),
        )
        vgg_remove_lists = ['fc6', 'fc7', 'fc8']
        for name in vgg_remove_lists:
            self.trunk._children.remove(name)
            delattr(self.trunk, name)

        self.proposal_target_layer = ProposalTargetLayer(num_classes)
        self.train = True
        self.gpu = gpu
        self.sigma = sigma

        self.spatial_scale = spatial_scale

    def __call__(self, x, bboxes=None):
        img_H, img_W = x.shape[2:]
        img_shape = (img_H, img_W)
        h = self.trunk(x, layers=['conv5_3'])['conv5_3']

        if self.train:
            bboxes.to_cpu()
            bboxes = bboxes.data
            rpn_cls_loss, rpn_loss_bbox, rois = self.RPN(
                h, img_shape, bboxes=bboxes)
        else:
            # shape (300, 5)
            # the second axis is (batch_id, x_min, y_min, x_max, y_max)
            rois = self.RPN(h, img_shape, bboxes=bboxes)

        if self.train:
            rois, labels, bbox_targets, bbox_inside_weights, \
                bbox_outside_weights = self.proposal_target_layer(
                    rois, bboxes)

        # Convert rois
        if self.gpu >= 0:
            rois = chainer.cuda.to_gpu(rois, device=self.gpu)
        boxes = rois[:, 1:5]

        # RCNN
        pool5 = roi_pooling_2d(
            h, rois, 7, 7, self.spatial_scale)
        fc6 = F.dropout(F.relu(self.fc6(pool5)), train=self.train)
        fc7 = F.dropout(F.relu(self.fc7(fc6)), train=self.train)

        # Per class probability
        cls_score = self.cls_score(fc7)
        cls_prob = F.softmax(cls_score)

        # BBox predictions
        bbox_pred = self.bbox_pred(fc7)
        box_deltas = bbox_pred.data

        if not self.train:
            pred_boxes = bbox_transform_inv(boxes, box_deltas, self.gpu)
            pred_boxes = clip_boxes(pred_boxes, img_shape, self.gpu)
            # returns NumPy arrays
            # (1, 300, 21) and (1, 300, 84)
            return cls_prob[None].data, pred_boxes[None]

        if self.gpu >= 0:
            def tg(x):
                return chainer.cuda.to_gpu(x, device=self.gpu)
            labels = tg(labels)
            bbox_targets = tg(bbox_targets)
            bbox_inside_weights = tg(bbox_inside_weights)
            bbox_outside_weights = tg(bbox_outside_weights)
        loss_cls = F.softmax_cross_entropy(cls_score, labels)
        labels = chainer.Variable(labels, volatile='auto')
        bbox_targets = chainer.Variable(bbox_targets, volatile='auto')
        loss_bbox = smooth_l1_loss(
            bbox_pred, bbox_targets, bbox_inside_weights,
            bbox_outside_weights, self.sigma)

        loss = rpn_cls_loss + rpn_loss_bbox + loss_bbox + loss_cls
        chainer.reporter.report({'rpn_loss_cls': rpn_cls_loss,
                                 'rpn_loss_bbox': rpn_loss_bbox,
                                 'loss_bbox': loss_bbox,
                                 'loss_cls': loss_cls,
                                 'loss': loss},
                                self)

        return loss
