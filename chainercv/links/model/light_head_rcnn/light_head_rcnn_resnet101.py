from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.functions import ps_roi_max_align_2d
from chainercv.links.connection.conv_2d_bn_activ import Conv2DBNActiv
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.light_head_rcnn.global_context_module import \
    GlobalContextModule
from chainercv.links.model.light_head_rcnn.light_head_rcnn import \
    LightHeadRCNN
from chainercv.links.model.resnet.resblock import ResBlock
from chainercv.links.model.resnet.resnet import ResNet101
from chainercv import utils


class LightHeadRCNNResNet101(LightHeadRCNN):

    """LightHead RCNN based on ResNet101.

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`coco`: Loads weights trained with the trainval split of \
        COCO Detection Dataset.
    * :obj:`imagenet`: Loads weights trained with ImageNet Classfication \
        task for the feature extractor and the head modules. \
        Weights that do not have a corresponding layer in ResNet101 \
        will be randomly initialized.

    For descriptions on the interface of this model, please refer to
    :class:`~light_head_rcnn.links.model.light_head_rcnn_base.LightHeadRCNN`

    :class:`~light_head_rcnn.links.model.light_head_rcnn_base.LightHeadRCNN`
    supports finer control on random initializations of weights by arguments
    :obj:`resnet_initialW`, :obj:`rpn_initialW`, :obj:`loc_initialW` and
    :obj:`score_initialW`.
    It accepts a callable that takes an array and edits its values.
    If :obj:`None` is passed as an initializer, the default initializer is
    used.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (string): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        min_size (int): A preprocessing paramter for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        resnet_initialW (callable): Initializer for the layers corresponding to
            the ResNet101 layers.
        rpn_initialW (callable): Initializer for Region Proposal Network
            layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.
        proposal_creator_params (dict): Key valued paramters for
            :class:`~chainercv.links.model.faster_rcnn.ProposalCreator`.

    """

    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': 'https://chainercv-models.preferred.jp/'
            'light_head_rcnn_resnet101_trained_2019_06_13.npz',
            'cv2': True
        },
    }
    feat_stride = 16
    proposal_creator_params = {
        'nms_thresh': 0.7,
        'n_train_pre_nms': 12000,
        'n_train_post_nms': 2000,
        'n_test_pre_nms': 6000,
        'n_test_post_nms': 1000,
        'force_cpu_nms': False,
        'min_size': 0,
    }

    def __init__(
            self,
            n_fg_class=None,
            pretrained_model=None,
            min_size=800, max_size=1333, roi_size=7,
            ratios=[0.5, 1, 2], anchor_scales=[2, 4, 8, 16, 32],
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
            resnet_initialW=None, rpn_initialW=None,
            global_module_initialW=None,
            loc_initialW=None, score_initialW=None,
            proposal_creator_params=None,
    ):

        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        if resnet_initialW is None and pretrained_model:
            resnet_initialW = chainer.initializers.HeNormal()
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if global_module_initialW is None:
            global_module_initialW = chainer.initializers.Normal(0.01)
        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if proposal_creator_params is not None:
            self.proposal_creator_params = proposal_creator_params

        extractor = ResNet101Extractor(
            initialW=resnet_initialW)
        rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=self.proposal_creator_params,
        )

        head = LightHeadRCNNResNet101Head(
            param['n_fg_class'] + 1,
            roi_size=roi_size,
            spatial_scale=1. / self.feat_stride,
            global_module_initialW=global_module_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW
        )
        mean = np.array([122.7717, 115.9465, 102.9801],
                        dtype=np.float32)[:, None, None]

        super(LightHeadRCNNResNet101, self).__init__(
            extractor, rpn, head, mean, min_size, max_size,
            loc_normalize_mean, loc_normalize_std)

        if path == 'imagenet':
            self._copy_imagenet_pretrained_resnet()
        elif path:
            chainer.serializers.load_npz(path, self)

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

    """ResNet101 Extractor for LightHeadRCNN ResNet101 implementation.

    This class is used as an extractor for LightHeadRCNNResNet101.
    This outputs feature maps.

    Args:
        initialW: Initializer for ResNet101 extractor.
    """

    def __init__(self, initialW=None):
        super(ResNet101Extractor, self).__init__()

        if initialW is None:
            initialW = chainer.initializers.HeNormal()
        kwargs = {
            'initialW': initialW,
            'bn_kwargs': {'eps': 1e-5, 'decay': 0.997},
            'stride_first': True
        }

        with self.init_scope():
            # ResNet
            self.conv1 = Conv2DBNActiv(
                3, 64, 7, 2, 3, nobias=True, initialW=initialW)
            self.pool1 = lambda x: F.max_pooling_2d(
                x, ksize=3, stride=2, pad=1, cover_all=False)
            self.res2 = ResBlock(3, 64, 64, 256, 1, **kwargs)
            self.res3 = ResBlock(4, 256, 128, 512, 2, **kwargs)
            self.res4 = ResBlock(23, 512, 256, 1024, 2, **kwargs)
            self.res5 = ResBlock(3, 1024, 512, 2048, 1, 2, **kwargs)

    def __call__(self, x):
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


class LightHeadRCNNResNet101Head(chainer.Chain):

    def __init__(
            self, n_class, roi_size, spatial_scale,
            global_module_initialW=None,
            loc_initialW=None, score_initialW=None
    ):

        super(LightHeadRCNNResNet101Head, self).__init__()
        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        with self.init_scope():
            self.global_context_module = GlobalContextModule(
                2048, 256, self.roi_size * self.roi_size * 10, 15,
                initialW=global_module_initialW)
            self.fc1 = L.Linear(
                self.roi_size * self.roi_size * 10, 2048,
                initialW=score_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)
            self.cls_loc = L.Linear(2048, 4 * n_class, initialW=loc_initialW)

    def __call__(self, x, rois, roi_indices):
        # global context module
        h = self.global_context_module(x)
        # psroi max align
        pool = ps_roi_max_align_2d(
            h, rois, roi_indices,
            (10, self.roi_size, self.roi_size),
            self.spatial_scale, self.roi_size,
            sampling_ratio=2)
        pool = F.where(
            self.xp.isinf(pool.array),
            self.xp.zeros(pool.shape, dtype=pool.dtype), pool)

        # fc
        fc1 = F.relu(self.fc1(pool))
        roi_cls_locs = self.cls_loc(fc1)
        roi_scores = self.score(fc1)
        return roi_cls_locs, roi_scores
