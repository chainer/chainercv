from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainercv.experimental.links.model.fcis import FCIS
from chainercv.functions import ps_roi_average_pooling_2d
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.links.model.resnet.resblock import ResBlock
from chainercv.links import ResNet101
from chainercv import utils


class FCISResNet101(FCIS):

    """FCIS based on ResNet101.

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`sbd`: Loads weights trained with the trainval split of Semantic \
    Boundaries Dataset.

    For descriptions on the interface of this model, please refer to
    :class:`~chainercv.experimental.links.model.fcis.FCIS`.

    :class:`~chainercv.experimental.links.model.fcis.FCISResNet101`
    supports finer control on random initializations of weights by arguments
    :obj:`resnet_initialW`, :obj:`rpn_initialW` and :obj:`head_initialW`.
    It accepts a callable that takes an array and edits its values.
    If :obj:`None` is passed as an initializer, the default initializer is
    used.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (str): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        min_size (int): A preprocessing paramter for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
        roi_size (int): Height and width of the feature maps after
            Position Sensitive RoI pooling.
        group_size (int): Group height and width for Position Sensitive
            ROI pooling.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.
        resnet_initialW (callable): Initializer for the layers corresponding to
            the ResNet101 layers.
        rpn_initialW (callable): Initializer for Region Proposal Network
            layers.
        head_initialW (callable): Initializer for the head layers.
        proposal_creator_params (dict): Key valued paramters for
            :class:`~chainercv.links.model.faster_rcnn.ProposalCreator`.

    """

    _models = {
        'sbd': {
            'url': 'https://chainercv-models.preferred.jp/'
                   'fcis_resnet101_sbd_trained_2018_06_22.npz',
            'preset_param': 'sbd',
            'cv2': True,
        },
        'sbd_converted': {
            'url': 'https://chainercv-models.preferred.jp/'
                   'fcis_resnet101_sbd_converted_2018_07_02.npz',
            'preset_param': 'sbd',
            'cv2': True,
        },
        'coco': {
            'url': 'https://chainercv-models.preferred.jp/'
                   'fcis_resnet101_coco_trained_2019_01_30.npz',
            'preset_param': 'coco',
            'cv2': True,
        },
        'coco_converted': {
            'url': 'https://chainercv-models.preferred.jp/'
                   'fcis_resnet101_coco_converted_2019_01_30.npz',
            'preset_param': 'coco',
            'cv2': True,
        },
    }
    _params = {
        'sbd': {
            'n_fg_class': 20,
            'anchor_scales': (8, 16, 32),
            'proposal_creator_params': {
                'nms_thresh': 0.7,
                'n_train_pre_nms': 6000,
                'n_train_post_nms': 300,
                'n_test_pre_nms': 6000,
                'n_test_post_nms': 300,
                'force_cpu_nms': False,
                'min_size': 16,
            },
        },
        'coco': {
            'n_fg_class': 80,
            'anchor_scales': (4, 8, 16, 32),
            'proposal_creator_params': {
                'nms_thresh': 0.7,
                'n_train_pre_nms': 6000,
                'n_train_post_nms': 300,
                'n_test_pre_nms': 6000,
                'n_test_post_nms': 300,
                'force_cpu_nms': False,
                'min_size': 2,
            },
        },
    }
    _default_param = {
        'feat_stride': 16,
        'min_size': 600,
        'max_size': 1000,
        'roi_size': 21,
        'group_size': 7,
        'ratios': [0.5, 1, 2],
        'loc_normalize_mean': (0.0, 0.0, 0.0, 0.0),
        'loc_normalize_std': (0.2, 0.2, 0.5, 0.5),
    }

    def __init__(
            self,
            n_fg_class=None,
            pretrained_model=None,
            feat_stride=None,
            min_size=None, max_size=None,
            roi_size=None, group_size=None,
            ratios=None, anchor_scales=None,
            loc_normalize_mean=None, loc_normalize_std=None,
            proposal_creator_params=None,
    ):

        path, preset_param = utils.prepare_pretrained_model(
            pretrained_model, self._models)
        param = utils.prepare_param(
            {
                'n_fg_class': n_fg_class,
                'feat_stride': feat_stride,
                'min_size': min_size,
                'max_size': max_size,
                'roi_size': roi_size,
                'group_size': group_size,
                'ratios': ratios,
                'anchor_scales': anchor_scales,
                'loc_normalize_mean': loc_normalize_mean,
                'loc_normalize_std': loc_normalize_std,
                'iter2': iter2,
                'proposal_creator_params': proposal_creator_params
            },
            self.preset_param(preset_param))

        rpn_initialW = chainer.initializers.Normal(0.01)
        resnet_initialW = chainer.initializers.constant.Zero()
        head_initialW = chainer.initializers.Normal(0.01)

        extractor = ResNet101Extractor(initialW=resnet_initialW)
        rpn = RegionProposalNetwork(
            1024, 512,
            ratios=param['ratios'],
            anchor_scales=param['anchor_scales'],
            feat_stride=param['feat_stride'],
            initialW=rpn_initialW,
            proposal_creator_params=param['proposal_creator_params'])
        head = FCISResNet101Head(
            param['n_fg_class'] + 1,
            roi_size=param['roi_size'], group_size=param['group_size'],
            spatial_scale=1. / param['feat_stride'],
            loc_normalize_mean=param['loc_normalize_mean'],
            loc_normalize_std=param['loc_normalize_std'],
            initialW=head_initialW)

        mean = np.array(
            [123.15, 115.90, 103.06], dtype=np.float32)[:, None, None]

        super(FCISResNet101, self).__init__(
            extractor, rpn, head,
            mean, param['min_size'], param['max_size'],
            param['loc_normalize_mean'], param['loc_normalize_std'])

        if path == 'imagenet':
            self._copy_imagenet_pretrained_resnet()
        elif path:
            chainer.serializers.load_npz(path, self)

    @classmethod
    def preset_param(cls, preset_param):
        if preset_param is None:
            return None
        param = cls._params[preset_param].copy()
        param = dict(param, **cls._default_param)
        return param

    def _copy_imagenet_pretrained_resnet(self):
        def _copy_conv2dbn(src, dst):
            dst.conv.W.array = src.conv.W.array
            if src.conv.b is not None and dst.conv.b is not None:
                dst.conv.b.array = src.conv.b.array
            dst.bn.gamma.array = src.bn.gamma.array
            dst.bn.beta.array = src.bn.beta.array
            dst.bn.avg_var = src.bn.avg_var
            dst.bn.avg_mean = src.bn.avg_mean

        def _copy_bottleneck(src, dst):
            if hasattr(src, 'residual_conv'):
                _copy_conv2dbn(src.residual_conv, dst.residual_conv)
            _copy_conv2dbn(src.conv1, dst.conv1)
            _copy_conv2dbn(src.conv2, dst.conv2)
            _copy_conv2dbn(src.conv3, dst.conv3)

        def _copy_resblock(src, dst):
            for layer_name in src.layer_names:
                _copy_bottleneck(
                    getattr(src, layer_name), getattr(dst, layer_name))

        pretrained_model = ResNet101(arch='he', pretrained_model='imagenet')
        _copy_conv2dbn(pretrained_model.conv1, self.extractor.conv1)
        _copy_resblock(pretrained_model.res2, self.extractor.res2)
        _copy_resblock(pretrained_model.res3, self.extractor.res3)
        _copy_resblock(pretrained_model.res4, self.extractor.res4)
        _copy_resblock(pretrained_model.res5, self.extractor.res5)


class ResNet101Extractor(chainer.Chain):

    """ResNet101 Extractor for FCIS ResNet101 implementation.

    This class is used as an extractor for FCISResNet101.
    This outputs feature maps.
    Dilated convolution is used in the C5 stage.

    Args:
        initialW: Initializer for ResNet101 extractor.
    """

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

    def forward(self, x):
        """Forward the chain.

        Args:
            x (~chainer.Variable): 4D image variable.

       """

        with chainer.using_config('train', False):
            h = self.pool1(self.conv1(x))
            h = self.res2(h)
            h.unchain_backward()
            h = self.res3(h)
            res4 = self.res4(h)
            res5 = self.res5(res4)
        return res4, res5


class FCISResNet101Head(chainer.Chain):

    """FCIS Head for ResNet101 based implementation.

    This class is used as a head for FCIS.
    This outputs class-agnostice segmentation scores, class-agnostic
    localizations and classification based on feature maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after
            Position Sensitive RoI pooling.
        group_size (int): Group height and width for Position Sensitive
            ROI pooling.
        spatial_scale (float): Scale of the roi is resized.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.
        initialW (callable): Initializer for the layers.

    """

    def __init__(
            self,
            n_class,
            roi_size, group_size, spatial_scale,
            loc_normalize_mean, loc_normalize_std,
            initialW=None
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

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                2048, 1024, 1, 1, 0, initialW=initialW)
            self.cls_seg = L.Convolution2D(
                1024, group_size * group_size * n_class * 2,
                1, 1, 0, initialW=initialW)
            self.ag_loc = L.Convolution2D(
                1024, group_size * group_size * 2 * 4,
                1, 1, 0, initialW=initialW)

    def forward(self, x, rois, roi_indices, img_size,
                 gt_roi_labels=None, iter2=False):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (~chainer.Variable): 4D image variable.
            rois (array): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (array): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
            img_size (tuple of int): A tuple containing image size.
            iter2 (bool): if the value is set :obj:`True`, Position Sensitive
                ROI pooling is executed twice. In the second time, Position
                Sensitive ROI pooling uses improved ROIs by the localization
                parameters calculated in the first time.

        """
        h = F.relu(self.conv1(x))
        h_cls_seg = self.cls_seg(h)
        h_ag_loc = self.ag_loc(h)

        # PSROI pooling and regression
        roi_ag_seg_scores, roi_ag_locs, roi_cls_scores = self._pool(
            h_cls_seg, h_ag_loc, rois, roi_indices, gt_roi_labels)
        if iter2:
            # 2nd Iteration
            # get rois2 for more precise prediction
            roi_ag_locs = roi_ag_locs.array
            mean = self.xp.array(self.loc_normalize_mean)
            std = self.xp.array(self.loc_normalize_std)
            roi_locs = roi_ag_locs[:, 1, :]
            roi_locs = (roi_locs * std + mean).astype(np.float32)
            rois2 = loc2bbox(rois, roi_locs)

            rois2[:, 0::2] = self.xp.clip(rois2[:, 0::2], 0, img_size[0])
            rois2[:, 1::2] = self.xp.clip(rois2[:, 1::2], 0, img_size[1])

            # PSROI pooling and regression
            roi_ag_seg_scores2, roi_ag_locs2, roi_cls_scores2 = self._pool(
                h_cls_seg, h_ag_loc, rois2, roi_indices, gt_roi_labels)

            # concat 1st and 2nd iteration results
            rois = self.xp.concatenate((rois, rois2))
            roi_indices = self.xp.concatenate((roi_indices, roi_indices))
            roi_ag_seg_scores = F.concat(
                (roi_ag_seg_scores, roi_ag_seg_scores2), axis=0)
            roi_ag_locs = F.concat(
                (roi_ag_locs, roi_ag_locs2), axis=0)
            roi_cls_scores = F.concat(
                (roi_cls_scores, roi_cls_scores2), axis=0)
        return roi_ag_seg_scores, roi_ag_locs, roi_cls_scores, \
            rois, roi_indices

    def _pool(
            self, h_cls_seg, h_ag_loc, rois, roi_indices, gt_roi_labels):
        # PSROI Pooling
        # shape: (n_roi, n_class, 2, roi_size, roi_size)
        roi_cls_ag_seg_scores = ps_roi_average_pooling_2d(
            h_cls_seg, rois, roi_indices,
            (self.n_class * 2, self.roi_size, self.roi_size),
            self.spatial_scale, self.group_size)
        roi_cls_ag_seg_scores = F.reshape(
            roi_cls_ag_seg_scores,
            (-1, self.n_class, 2, self.roi_size, self.roi_size))

        # shape: (n_roi, 2*4, roi_size, roi_size)
        roi_ag_loc_scores = ps_roi_average_pooling_2d(
            h_ag_loc, rois, roi_indices,
            (2 * 4, self.roi_size, self.roi_size),
            self.spatial_scale, self.group_size)

        # shape: (n_roi, n_class)
        roi_cls_scores = F.average(
            F.max(roi_cls_ag_seg_scores, axis=2), axis=(2, 3))

        # Bbox Regression
        # shape: (n_roi, 2, 4)
        roi_ag_locs = F.average(roi_ag_loc_scores, axis=(2, 3))
        roi_ag_locs = F.reshape(roi_ag_locs, (-1, 2, 4))

        # Mask Regression
        # shape: (n_roi, n_class, 2, roi_size, roi_size)
        if gt_roi_labels is None:
            max_cls_indices = roi_cls_scores.array.argmax(axis=1)
        else:
            max_cls_indices = gt_roi_labels

        # shape: (n_roi, 2, roi_size, roi_size)
        roi_ag_seg_scores = roi_cls_ag_seg_scores[
            self.xp.arange(len(max_cls_indices)), max_cls_indices]

        return roi_ag_seg_scores, roi_ag_locs, roi_cls_scores
