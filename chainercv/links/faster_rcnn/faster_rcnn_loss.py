import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.faster_rcnn.region_proposal_network import\
    RegionProposalNetworkLoss

from chainercv.functions.smooth_l1_loss import smooth_l1_loss
from chainercv.links.faster_rcnn.proposal_target_layer import\
    ProposalTargetLayer


class FasterRCNNLoss(chainer.Chain):

    def __init__(self, faster_rcnn, rpn_sigma=3., sigma=1.,
                 anchor_target_layer_params={},
                 proposal_target_layer_params={},
                 ):
        self.sigma = sigma
        self.n_class = faster_rcnn.n_class
        self.roi_size = faster_rcnn.roi_size
        self.spatial_scale = faster_rcnn.spatial_scale
        super(FasterRCNNLoss, self).__init__(faster_rcnn=faster_rcnn)

        # These paramteres need to be consistent across modules
        proposal_target_layer_params.update({
            'n_class': self.n_class,
            'bbox_normalize_target_precomputed': self.faster_rcnn.target_precomputed,
            'bbox_normalize_mean': self.faster_rcnn.bbox_normalize_mean,
            'bbox_normalize_std': self.faster_rcnn.bbox_normalize_std,
        })
        self.proposal_target_layer = ProposalTargetLayer(
            **proposal_target_layer_params)
        self.rpn_loss_func = RegionProposalNetworkLoss(
            rpn_sigma, anchor_target_layer_params)
        self.train = True

    def __call__(self, img, bbox, label, scale=1.):
        # TODO(yuyu2172) use train setting for feature extraction as well
        # some layers need the parameter (like ResNet)
        img_size = img.shape[2:][::-1]
        layers = ['feature', 'rpn_bbox_pred', 'rpn_cls_score',
                  'roi', 'anchor']
        out = self.faster_rcnn(
            img, scale=scale, layers=layers, rpn_only=True,
            test=not self.train)

        roi_sample, label_sample, bbox_target_sample, bbox_inside_weight, \
            bbox_outside_weight = self.proposal_target_layer(
                out['roi'], bbox, label)
        # forward sampled RoIs
        pool5 = F.roi_pooling_2d(
            out['feature'],
            roi_sample, self.roi_size, self.roi_size, self.spatial_scale)
        bbox_tf, cls_score = self.faster_rcnn.head(pool5, train=self.train)

        # Losses for outputs of the head.
        loss_cls = F.softmax_cross_entropy(cls_score, label_sample)
        loss_bbox = smooth_l1_loss(
            bbox_tf, bbox_target_sample,
            bbox_inside_weight, bbox_outside_weight, self.sigma)

        # rpn losses
        rpn_loss_bbox, rpn_cls_loss = self.rpn_loss_func(
            out['rpn_bbox_pred'], out['rpn_cls_score'],
            out['roi'], bbox, out['anchor'],
            img_size)

        loss = rpn_loss_bbox + rpn_cls_loss + loss_bbox + loss_cls
        chainer.reporter.report({'rpn_loss_cls': rpn_cls_loss,
                                 'rpn_loss_bbox': rpn_loss_bbox,
                                 'loss_bbox': loss_bbox,
                                 'loss_cls': loss_cls,
                                 'loss': loss},
                                self)
        return loss
