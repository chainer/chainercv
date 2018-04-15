from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainercv.experimental.links.model.fcis import FCIS
from chainercv.functions import psroi_pooling_2d
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.links.model.resnet.resblock import ResBlock
from chainercv.utils import download_model


class FCISResNet101(FCIS):

    _models = {
        'sbd': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.6/fcis_resnet101_2018_04_14.npz'
        }
    }
    feat_stride = 16

    def __init__(
            self,
            n_fg_class=None,
            pretrained_model=None,
            min_size=600, max_size=1000,
            ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
            loc_normalize_mean=(0.0, 0.0, 0.0, 0.0),
            loc_normalize_std=(0.2, 0.2, 0.5, 0.5),
            iter2=True,
            resnet_initialW=None, rpn_initialW=None, head_initialW=None,
            proposal_creator_params={
                'nms_thresh': 0.7,
                'n_train_pre_nms': 6000,
                'n_train_post_nms': 300,
                'n_test_pre_nms': 6000,
                'n_test_post_nms': 300,
                'force_cpu_nms': False,
                'min_size': 16
            }
    ):
        if n_fg_class is None:
            if pretrained_model not in self._models:
                raise ValueError(
                    'The n_fg_class needs to be supplied as an argument')
            n_fg_class = self._models[pretrained_model]['n_fg_class']

        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if resnet_initialW is None and pretrained_model:
            resnet_initialW = chainer.initializers.constant.Zero()

        extractor = ResNet101Extractor(
            initialW=resnet_initialW)
        rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params)
        head = FCISResNet101Head(
            n_fg_class + 1,
            roi_size=21, group_size=7,
            spatial_scale=1. / self.feat_stride,
            loc_normalize_mean=loc_normalize_mean,
            loc_normalize_std=loc_normalize_std,
            iter2=iter2, initialW=head_initialW)

        mean = np.array([123.15, 115.90, 103.06],
                        dtype=np.float32)[:, None, None]

        super(FCISResNet101, self).__init__(
            extractor, rpn, head,
            mean, min_size, max_size,
            loc_normalize_mean, loc_normalize_std)

        if pretrained_model is not None:
            if pretrained_model in self._models:
                path = download_model(self._models[pretrained_model]['url'])
            else:
                path = pretrained_model
            chainer.serializers.load_npz(path, self)


class ResNet101Extractor(chainer.Chain):

    def __init__(self, initialW=None):
        super(ResNet101Extractor, self).__init__()

        if initialW is None:
            initialW = chainer.initializers.HeNormal()
        kwargs = {
            'initialW': initialW,
            'bn_kwargs': {'eps': 1e-5},
            'stride_first': True
        }

        with self.init_scope():
            # ResNet
            self.conv1 = Conv2DBNActiv(
                3, 64, 7, 2, 3, nobias=True, initialW=initialW)
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.res2 = ResBlock(3, 64, 64, 256, 1, **kwargs)
            self.res3 = ResBlock(4, 256, 128, 512, 2, **kwargs)
            self.res4 = ResBlock(23, 512, 256, 1024, 2, **kwargs)
            self.res5 = ResBlock(3, 1024, 512, 2048, 1, 2, **kwargs)

    def __call__(self, x):
        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                h = self.pool1(self.conv1(x))
                h = self.res2(h)
            h = self.res3(h)
            res4 = self.res4(h)
            res5 = self.res5(res4)
        return res4, res5


class FCISResNet101Head(chainer.Chain):

    def __init__(
            self,
            n_class,
            roi_size, group_size, spatial_scale,
            loc_normalize_mean, loc_normalize_std,
            iter2, initialW=None
    ):
        super(FCISResNet101Head, self).__init__()

        if initialW is None:
            initialW = chainer.initializers.Normal(0.01)

        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.group_size = group_size
        self.roi_size = roi_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.iter2 = iter2

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                2048, 1024, 1, 1, 0, initialW=initialW)
            self.cls_seg = L.Convolution2D(
                1024, group_size * group_size * n_class * 2,
                1, 1, 0, initialW=initialW)
            self.ag_loc = L.Convolution2D(
                1024, group_size * group_size * 2 * 4,
                1, 1, 0, initialW=initialW)

    def __call__(self, x, rois, roi_indices, img_size):
        h = F.relu(self.conv1(x))
        h_cls_seg = self.cls_seg(h)
        h_ag_loc = self.ag_loc(h)

        # PSROI pooling and regression
        roi_seg_scores, roi_ag_locs, roi_cls_scores = self._pool(
            h_cls_seg, h_ag_loc, rois, roi_indices)
        if self.iter2:
            # 2nd Iteration
            # get rois2 for more precise prediction
            roi_ag_locs = roi_ag_locs.array
            mean = self.xp.array(self.loc_normalize_mean)
            std = self.xp.array(self.loc_normalize_std)
            roi_loc = roi_ag_locs[:, 1, :]
            roi_loc = (roi_loc * std + mean).astype(np.float32)
            rois2 = loc2bbox(rois, roi_loc)

            rois2[:, 0::2] = self.xp.clip(rois2[:, 0::2], 0, img_size[0])
            rois2[:, 1::2] = self.xp.clip(rois2[:, 1::2], 0, img_size[1])

            # PSROI pooling and regression
            roi_seg_scores2, roi_ag_locs2, roi_cls_scores2 = self._pool(
                h_cls_seg, h_ag_loc, rois2, roi_indices)

            # concat 1st and 2nd iteration results
            rois = self.xp.concatenate((rois, rois2))
            roi_indices = self.xp.concatenate((roi_indices, roi_indices))
            roi_seg_scores = F.concat(
                (roi_seg_scores, roi_seg_scores2), axis=0)
            roi_ag_locs = F.concat(
                (roi_ag_locs, roi_ag_locs2), axis=0)
            roi_cls_scores = F.concat(
                (roi_cls_scores, roi_cls_scores2), axis=0)
        return roi_seg_scores, roi_ag_locs, roi_cls_scores, rois, roi_indices

    def _pool(
            self, h_cls_seg, h_ag_loc, rois, roi_indices):
        # PSROI Pooling
        # shape: (n_roi, n_class*2, roi_size, roi_size)
        roi_cls_seg_scores = psroi_pooling_2d(
            h_cls_seg, rois, roi_indices,
            self.n_class * 2, self.roi_size, self.roi_size,
            self.spatial_scale, self.group_size)
        roi_cls_seg_scores = roi_cls_seg_scores.reshape(
            (-1, self.n_class, 2, self.roi_size, self.roi_size))

        # shape: (n_roi, 2*4, roi_size, roi_size)
        roi_ag_loc_scores = psroi_pooling_2d(
            h_ag_loc, rois, roi_indices,
            2 * 4, self.roi_size, self.roi_size,
            self.spatial_scale, self.group_size)

        # shape: (n_roi, n_class)
        roi_cls_scores = _global_average_pooling_2d(
            F.max(roi_cls_seg_scores, axis=2))

        # Bbox Regression
        # shape: (n_roi, 2*4)
        roi_ag_locs = _global_average_pooling_2d(roi_ag_loc_scores)
        roi_ag_locs = roi_ag_locs.reshape((-1, 2, 4))

        # Mask Regression
        # shape: (n_roi, n_class, 2, roi_size, roi_size)
        max_cls_indices = roi_cls_scores.array.argmax(axis=1)
        # shape: (n_roi, 2, roi_size, roi_size)
        roi_seg_scores = roi_cls_seg_scores[
            self.xp.arange(len(max_cls_indices)), max_cls_indices]

        return roi_seg_scores, roi_ag_locs, roi_cls_scores


def _global_average_pooling_2d(x):
    n_roi, n_channel, H, W = x.array.shape
    h = F.average_pooling_2d(x, (H, W), stride=1)
    h = F.reshape(h, (n_roi, n_channel))
    return h
