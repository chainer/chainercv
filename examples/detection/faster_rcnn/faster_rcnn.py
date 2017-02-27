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

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links.model.vision.vgg import VGG16Layers

from lib.bbox_transform import bbox_transform_inv
from lib.bbox_transform import clip_boxes
from lib.nms_cpu import nms_cpu as nms
from lib.proposal_target_layer import ProposalTargetLayer
from lib.rpn import RPN
from lib.roi_pooling_2d import roi_pooling_2d
from lib.smooth_l1_loss import smooth_l1_loss


class FasterRCNN(chainer.Chain):

    def __init__(
            self, gpu=-1, rpn_in_ch=512, rpn_out_ch=512,
            n_anchors=9, feat_stride=16, anchor_scales=[8, 16, 32],
            num_classes=21, spatial_scale=0.0625, rpn_sigma=1.0, sigma=3.0,
            nms_thresh=0.3, confidence=0.8
    ):
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

        self.nms_thresh = 0.3
        self.confidence = 0.8

    def __call__(self, x, bboxes=None):
        img_H, img_W = x.shape[2:]
        img_shape = (img_H, img_W)
        h = self.trunk(x, layers=['conv5_3'])['conv5_3']

        if self.train:
            bboxes = bboxes[:1]  # TODO(yuyu2172) fix
            bboxes.to_cpu()
            bboxes = bboxes.data
            rpn_cls_loss, rpn_loss_bbox, rois = self.RPN(
                h, img_shape, bboxes=bboxes, gpu=self.gpu)
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
            # returns arrays
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

        return rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls
        # return loss

    def predict_bboxes(self, x, bboxes=None):
        """Predicts bounding boxes which satisfy confidence constraints.

        """
        cls_prob, pred_bboxes = self.__call__(x, bboxes=None)
        cls_prob = chainer.cuda.to_cpu(cls_prob)
        pred_bboxes = chainer.cuda.to_cpu(pred_bboxes)

        final_bboxes = _predict_to_bboxes(
            cls_prob[0], pred_bboxes[0], self.nms_thresh, self.confidence)
        return final_bboxes[None]


def _predict_to_bboxes(cls_prob, pred_bboxes, nms_thresh, confidence):
    final_bboxes = []
    for cls_id in range(1, 21):
        _cls = cls_prob[:, cls_id][:, None]  # (300, 1)
        _bbx = pred_bboxes[:, cls_id * 4: (cls_id + 1) * 4]  # (300, 4)
        dets = np.hstack((_bbx, _cls))  # (300, 5)
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= confidence)[0]
        if len(inds) > 0:
            selected = dets[inds]
            final_bboxes.append(
                np.concatenate(
                    (selected[:, :4], np.ones((len(selected), 1)) * cls_id),
                    axis=1)
            )
    if len(final_bboxes) != 0:
        final_bboxes = np.concatenate(final_bboxes, axis=0)
    else:
        final_bboxes = np.zeros((0, 5), dtype=np.float32)
    return final_bboxes
