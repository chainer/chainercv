from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet import ResBlock
from chainercv.links import PickableSequentialChain
from chainercv import utils


class PyramidPoolingModule(chainer.ChainList):

    def __init__(self, in_channels, feat_size, pyramids,
                 initialW=None,
                 bn_kwargs=None):
        out_channels = in_channels // len(pyramids)
        super(PyramidPoolingModule, self).__init__(
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs)
        )
        kh = feat_size[0] // np.array(pyramids)
        kw = feat_size[1] // np.array(pyramids)
        self.ksizes = list(zip(kh, kw))

    def __call__(self, x):
        ys = [x]
        H, W = x.shape[2:]
        for f, ksize in zip(self, self.ksizes):
            y = F.average_pooling_2d(x, ksize, ksize)
            y = f(y)
            y = F.resize_images(y, (H, W))
            ys.append(y)
        return F.concat(ys, axis=1)


class DilatedResNet(PickableSequentialChain):

    _blocks = {
        101: [3, 4, 23, 3],
    }

    def __init__(self, n_layer, initialW, bn_kwargs=None):
        n_block = self._blocks[n_layer]
        super(DilatedResNet, self).__init__()
        with self.init_scope():
            self.conv1_1 = Conv2DBNActiv(
                None, 64, 3, 2, 1, 1,
                initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv1_2 = Conv2DBNActiv(
                64, 64, 3, 1, 1, 1, initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv1_3 = Conv2DBNActiv(
                64, 128, 3, 1, 1, 1, initialW=initialW, bn_kwargs=bn_kwargs)
            self.res2 = ResBlock(
                n_block[0], 128, 64, 256, 1, 1,
                initialW, bn_kwargs, stride_first=False)
            self.res3 = ResBlock(
                n_block[1], 256, 128, 512, 2, 1,
                initialW, bn_kwargs, stride_first=False)
            self.res4 = ResBlock(
                n_block[2], 512, 256, 1024, 1, 2,
                initialW, bn_kwargs, stride_first=False)
            self.res5 = ResBlock(
                n_block[3], 1024, 512, 2048, 1, 4,
                initialW, bn_kwargs, stride_first=False)


class PSPNet(chainer.Chain):

    """Pyramid Scene Parsing Network.

    This is a PSPNet [#]_ model for semantic segmentation. This is based on
    the implementation found here_.

    .. [#] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang \
    Jiaya Jia "Pyramid Scene Parsing Network" \
    CVPR, 2017

    .. _here: https://github.com/hszhao/PSPNet

    Args:
        n_layer (int): The number of layers.
        n_class (int): The number of channels in the last convolution layer.
        input_size (tuple): The size of the input.
            This value is :math:`(height, width)`.
        initialW (callable): Initializer for the weights of
            convolution kernels.
        bn_kwargs (dict): Keyword arguments passed to initialize
            :class:`chainer.links.BatchNormalization`. If a ChainerMN
            communicator (:class:`~chainermn.communicators.CommunicatorBase`)
            is given with the key :obj:`comm`,
            :obj:`~chainermn.links.MultiNodeBatchNormalization` will be used
            for the batch normalization. Otherwise,
            :obj:`~chainer.links.BatchNormalization` will be used.

    """

    def __init__(self, n_layer, n_class, input_size,
                 initialW=None, bn_kwargs=None):
        super(PSPNet, self).__init__()
        if initialW is None:
            initialW = chainer.initializers.HeNormal()

        pyramids = [6, 3, 2, 1]

        if not isinstance(input_size, (list, tuple)):
            input_size = (int(input_size), int(input_size))

        self.scales = None
        self.mean = np.array(
            [123.68, 116.779, 103.939], dtype=np.float32)[:, None, None]
        self.input_size = input_size

        feat_size = (input_size[0] // 8, input_size[1] // 8)
        with self.init_scope():
            self.extractor = DilatedResNet(n_layer=n_layer, initialW=initialW,
                                           bn_kwargs=bn_kwargs)
            self.extractor.pick = ('res4', 'res5')
            self.ppm = PyramidPoolingModule(2048, feat_size, pyramids,
                                            initialW=initialW,
                                            bn_kwargs=bn_kwargs)
            self.head_conv1 = Conv2DBNActiv(4096, 512, 3, 1, 1,
                                            initialW=initialW)
            self.head_conv2 = L.Convolution2D(
                512, n_class, 1, 1, 0, False, initialW)

    @property
    def n_class(self):
        return self.head_conv2.out_channels

    def __call__(self, x):
        _, res5 = self.extractor(x)

        h = self.ppm(res5)
        h = self.head_conv1(h)
        h = self.head_conv2(h)
        h = F.resize_images(h, x.shape[2:])
        return h

    def predict(self, imgs):
        """Conduct semantic segmentation from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.

        Returns:
            list of numpy.ndarray:

            List of integer labels predicted from each image in the input \
            list.

        """
        return utils.semantic_segmentation_predict(
            self.__call__,
            imgs, self.scales, self.mean, self.input_size, self.n_class,
            self.xp)


class PSPNetResNet101(PSPNet):

    """PSPNet with Dilated ResNet101 as the feature extractor.

    .. seealso::
        :class:`chainercv.experimental.links.model.pspnet.PSPNet`

    Args:
        n_class (int): The number of channels in the last convolution layer.
        pretrained_model (string): The weight file to be loaded.
            This can take :obj:`'cityscapes'`, `filepath` or :obj:`None`.
            The default value is :obj:`None`.

            * :obj:`'cityscapes'`: Load weights trained on train split of \
                Cityscapes dataset. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_class` must be :obj:`19` or :obj:`None`. \
            * `filepath`: A path of npz file. In this case, :obj:`n_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

        input_size (tuple): The size of the input.
            This value is :math:`(height, width)`.
        initialW (callable): Initializer for the weights of
            convolution kernels.
        comm (chainermn.communicator): If a ChainerMN communicator is
            given, it will be used for distributed batch normalization during
            training. If None, all batch normalization links will not share
            the input vectors among GPUs before calculating mean and variance.
            The original PSPNet implementation uses distributed batch
            normalization.

    """

    _models = {
        'cityscapes': {
            'param': {'n_class': 19, 'input_size': (713, 713)},
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.6/pspnet_resnet101_cityscapes_convert_2018_05_22.npz'
        }
    }

    def __init__(self, n_class=None, pretrained_model=None,
                 input_size=None,
                 initialW=None, comm=None):
        param, path = utils.prepare_pretrained_model(
            {'n_class': n_class, 'input_size': input_size},
            pretrained_model, self._models,
            {'input_size': (713, 713)})

        if comm is not None:
            bn_kwargs = {'comm': comm}
        else:
            bn_kwargs = {}
        super(PSPNetResNet101, self).__init__(
            101, param['n_class'], param['input_size'],
            initialW, bn_kwargs)

        if path:
            chainer.serializers.load_npz(path, self)
