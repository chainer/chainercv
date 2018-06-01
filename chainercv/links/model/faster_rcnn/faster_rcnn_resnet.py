import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.resnet.resblock import ResBlock
from chainercv.links.model.resnet.resnet import imagenet_he_mean

from chainercv.links.model.faster_rcnn.faster_rcnn import FasterRCNN
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.resnet import ResNet101
from chainercv.utils import download_model


def copy_persistent_link(dst, src):
    for name in dst._persistent:
        d = dst.__dict__[name]
        s = src.__dict__[name]
        if isinstance(d, np.ndarray):
            d[:] = s
        elif isinstance(d, int):
            d = s
        else:
            raise ValueError


def copy_persistent_chain(dst, src):
    copy_persistent_link(dst, src)
    for name in dst._children:
        if (isinstance(dst.__dict__[name], chainer.Chain) and
                isinstance(src.__dict__[name], chainer.Chain)):
            copy_persistent_chain(dst.__dict__[name], src.__dict__[name])
        elif (isinstance(dst.__dict__[name], chainer.Link) and
                isinstance(src.__dict__[name], chainer.Link)):
            copy_persistent_link(dst.__dict__[name], src.__dict__[name])


class FasterRCNNResNet101(FasterRCNN):

    """Faster R-CNN based on ResNet101.

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`voc07`: Loads weights trained with the trainval split of \
        PASCAL VOC2007 Detection Dataset.
    * :obj:`'voc0712'`: Load weights trained on trainval split of \
        PASCAL VOC 2007 and 2012. \
    * :obj:`imagenet`: Loads weights trained with ImageNet Classfication \
        task for the feature extractor and the head modules. \
        Weights that do not have a corresponding layer in ResNet101 \
        will be randomly initialized.

    For descriptions on the interface of this model, please refer to
    :class:`chainercv.links.model.faster_rcnn.FasterRCNN`.

    :obj:`FasterRCNNResNet101` supports finer control on
    random initializations of weights by arguments
    :obj:`res_initialW`, :obj:`rpn_initialW`, :obj:`loc_initialW` and
    :obj:`score_initialW`.
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
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        res_initialW (callable): Initializer for the layers corresponding to
            the ResNet101 layers.
        rpn_initialW (callable): Initializer for Region Proposal Network
            layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.
        proposal_creator_params (dict): Key valued paramters for
            :obj:`chainercv.links.model.faster_rcnn.ProposalCreator`.

    """

    _models = {}
    feat_stride = 16

    def __init__(self,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600, max_size=1000,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 res_initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params={}
                 ):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if res_initialW is None and pretrained_model:
            res_initialW = chainer.initializers.constant.Zero()

        extractor = ResNet101FeatureExtractor(res_initialW)
        rpn = RegionProposalNetwork(
            1024, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = ResNetRoIHead(
            n_fg_class + 1,
            roi_size=7, spatial_scale=1. / self.feat_stride,
            res_initialW=res_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW
        )

        super(FasterRCNNResNet101, self).__init__(
            extractor,
            rpn,
            head,
            mean=imagenet_he_mean,
            min_size=min_size,
            max_size=max_size
        )

        if path == 'imagenet':
            self._copy_imagenet_pretrained_resnet101()
        elif path:
            chainer.serializers.load_npz(path, self)

    def _copy_imagenet_pretrained_resnet101(self):
        pretrained_model = ResNet101(pretrained_model='imagenet', arch='he')
        dst_extractor = self.extractor.resnet

        dst_extractor.conv1.copyparams(pretrained_model.conv1)
        copy_persistent_chain(dst_extractor.conv1, pretrained_model.conv1)

        dst_extractor.res2.copyparams(pretrained_model.res2)
        copy_persistent_chain(dst_extractor.res2, pretrained_model.res2)

        dst_extractor.res3.copyparams(pretrained_model.res3)
        copy_persistent_chain(dst_extractor.res3, pretrained_model.res3)

        dst_extractor.res4.copyparams(pretrained_model.res4)
        copy_persistent_chain(dst_extractor.res4, pretrained_model.res4)

        self.head.res5.copyparams(pretrained_model.res5)
        copy_persistent_chain(self.head.res5, pretrained_model.res5)


class ResNetRoIHead(chainer.Chain):

    """Faster R-CNN Head for ResNet based implementation.

    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        res_initialW (callable): Initializer for the layers corresponding to
            the ResNet layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 res_initialW=None, loc_initialW=None, score_initialW=None):
        # n_class includes the background
        super(ResNetRoIHead, self).__init__()
        with self.init_scope():
            self.res5 = ResBlock(3, None, 512, 2048, 2,
                                 initialW=res_initialW)
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois, roi_indices):
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
            test (bool): Whether in test mode or not. This has no effect in
                the current implementation.

        """
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, np.newaxis], rois), axis=1)
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale)

        # Use train mode to stop updating batch statistics
        with chainer.using_config('train', False):
            h = self.res5(pool)
        h = F.max_pooling_2d(h, ksize=7)
        roi_cls_locs = self.cls_loc(h)
        roi_scores = self.score(h)
        return roi_cls_locs, roi_scores


class ResNet101FeatureExtractor(chainer.Chain):
    """Truncated ResNet101 that extracts a res4 feature map.

    Args:
        initialW (callable): Initializer for the weights.

    """

    def __init__(self, initialW=None):
        super(ResNet101FeatureExtractor, self).__init__()
        with self.init_scope():
            self.resnet = ResNet101(
                initialW=initialW,
                fc_kwargs={'initialW': chainer.initializers.constant.Zero()},
                arch='he')
        self.resnet.pick = 'res4'
        self.resnet.remove_unused()

    def __call__(self, x, test=True):
        # Use train mode to stop updating batch statistics
        with chainer.using_config('train', False):
            h = self.resnet(x)
        return h


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool
