from __future__ import division

from math import ceil
import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.experimental.links.model.pspnet.transforms import \
    convolution_crop
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet import ResBlock
from chainercv.links import PickableSequentialChain
from chainercv import transforms
from chainercv import utils


_imagenet_mean = np.array(
    (123.68, 116.779, 103.939), dtype=np.float32)[:, None, None]


class PyramidPoolingModule(chainer.ChainList):

    def __init__(self, in_channels, feat_size, pyramids, initialW=None):
        out_channels = in_channels // len(pyramids)
        super(PyramidPoolingModule, self).__init__(
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW),
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
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }

    _models = {
        50: {
            'imagenet': {
                'url': 'https://chainercv-models.preferred.jp/'
                'pspnet_resnet50_imagenet_trained_2018_11_26.npz',
                'cv2': True
            },
        },
        101: {
            'imagenet': {
                'url': 'https://chainercv-models.preferred.jp/'
                'pspnet_resnet101_imagenet_trained_2018_11_26.npz',
                'cv2': True
            },
        },
    }

    def __init__(self, n_layer, pretrained_model=None,
                 initialW=None):
        n_block = self._blocks[n_layer]

        _, path = utils.prepare_pretrained_model(
            {},
            pretrained_model,
            self._models[n_layer])

        super(DilatedResNet, self).__init__()
        with self.init_scope():
            self.conv1_1 = Conv2DBNActiv(
                None, 64, 3, 2, 1, 1, initialW=initialW)
            self.conv1_2 = Conv2DBNActiv(
                64, 64, 3, 1, 1, 1, initialW=initialW)
            self.conv1_3 = Conv2DBNActiv(
                64, 128, 3, 1, 1, 1, initialW=initialW)
            self.pool1 = lambda x: F.max_pooling_2d(
                x, ksize=3, stride=2, pad=1)
            self.res2 = ResBlock(
                n_block[0], 128, 64, 256, 1, 1,
                initialW=initialW, stride_first=False)
            self.res3 = ResBlock(
                n_block[1], 256, 128, 512, 2, 1,
                initialW=initialW, stride_first=False)
            self.res4 = ResBlock(
                n_block[2], 512, 256, 1024, 1, 2,
                initialW=initialW, stride_first=False)
            self.res5 = ResBlock(
                n_block[3], 1024, 512, 2048, 1, 4,
                initialW=initialW, stride_first=False)

        if path:
            chainer.serializers.load_npz(path, self, ignore_names=None)


class PSPNet(chainer.Chain):

    """Pyramid Scene Parsing Network.

    This is a PSPNet [#]_ model for semantic segmentation. This is based on
    the implementation found here_.

    .. [#] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang \
    Jiaya Jia "Pyramid Scene Parsing Network" \
    CVPR, 2017

    .. _here: https://github.com/hszhao/PSPNet

    Args:
        n_class (int): The number of channels in the last convolution layer.
        pretrained_model (string): The weight file to be loaded.
            This can take :obj:`'cityscapes'`, `filepath` or :obj:`None`.
            The default value is :obj:`None`.
            * :obj:`'cityscapes'`: Load weights trained on the train split of \
                Cityscapes dataset. \
                :obj:`n_class` must be :obj:`19` or :obj:`None`.
            * :obj:`'ade20k'`: Load weights trained on the train split of \
                ADE20K dataset. \
                :obj:`n_class` must be :obj:`150` or :obj:`None`.
            * :obj:`'imagenet'`: Load ImageNet pretrained weights for \
                the extractor.
            * `filepath`: A path of npz file. In this case, :obj:`n_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.
        input_size (tuple): The size of the input.
            This value is :math:`(height, width)`.
        initialW (callable): Initializer for the weights of
            convolution kernels.

    """

    def __init__(self, n_class=None, pretrained_model=None,
                 input_size=None, initialW=None):
        super(PSPNet, self).__init__()

        if pretrained_model == 'imagenet':
            extractor_pretrained_model = 'imagenet'
            pretrained_model = None
        else:
            extractor_pretrained_model = None

        param, path = utils.prepare_pretrained_model(
            {'n_class': n_class, 'input_size': input_size},
            pretrained_model, self._models,
            default={'input_size': (713, 713)})
        n_class = param['n_class']
        input_size = param['input_size']
        if not isinstance(input_size, (list, tuple)):
            input_size = (int(input_size), int(input_size))
        self.input_size = input_size

        if initialW is None:
            if pretrained_model:
                initialW = initializers.constant.Zero()

        kwargs = self._extractor_kwargs
        kwargs.update({'pretrained_model': extractor_pretrained_model,
                       'initialW': initialW})
        extractor = self._extractor_cls(**kwargs)
        extractor.pick = self._extractor_pick

        self.scales = None
        self.mean = _imagenet_mean

        pyramids = [6, 3, 2, 1]
        feat_size = (input_size[0] // 8, input_size[1] // 8)
        with self.init_scope():
            self.extractor = extractor
            self.ppm = PyramidPoolingModule(
                2048, feat_size, pyramids, initialW=initialW)
            self.head_conv1 = Conv2DBNActiv(
                4096, 512, 3, 1, 1, initialW=initialW)
            self.head_conv2 = L.Convolution2D(
                512, n_class, 1, 1, 0, False, initialW)

        if path:
            chainer.serializers.load_npz(path, self)

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

    def _tile_predict(self, img):
        if self.mean is not None:
            img = img - self.mean
        ori_H, ori_W = img.shape[1:]
        long_size = max(ori_H, ori_W)

        if long_size > max(self.input_size):
            stride_rate = 2 / 3
            stride = (int(ceil(self.input_size[0] * stride_rate)),
                      int(ceil(self.input_size[1] * stride_rate)))

            imgs, param = convolution_crop(
                img, self.input_size, stride, return_param=True)

            counts = self.xp.zeros((1, ori_H, ori_W), dtype=np.float32)
            preds = self.xp.zeros((1, self.n_class, ori_H, ori_W),
                                  dtype=np.float32)
            N = len(param['y_slices'])
            for i in range(N):
                img_i = imgs[i:i+1]
                y_slice = param['y_slices'][i]
                x_slice = param['x_slices'][i]
                crop_y_slice = param['crop_y_slices'][i]
                crop_x_slice = param['crop_x_slices'][i]

                scores_i = self._predict(img_i)
                # Flip horizontally flipped score maps again
                flipped_scores_i = self._predict(
                    img_i[:, :, :, ::-1])[:, :, :, ::-1]

                preds[0, :, y_slice, x_slice] +=\
                    scores_i[0, :, crop_y_slice, crop_x_slice]
                preds[0, :, y_slice, x_slice] +=\
                    flipped_scores_i[0, :, crop_y_slice, crop_x_slice]
                counts[0, y_slice, x_slice] += 2

            scores = preds / counts[:, None]
        else:
            img, param = transforms.resize_contain(
                img, self.input_size, return_param=True)
            preds1 = self._predict(img[np.newaxis])
            preds2 = self._predict(img[np.newaxis, :, :, ::-1])
            preds = (preds1 + preds2[:, :, :, ::-1]) / 2

            y_start = param['y_offset']
            y_end = y_start + param['scaled_size'][0]
            x_start = param['x_offset']
            x_end = x_start + param['scaled_size'][1]
            scores = preds[:, :, y_start:y_end, x_start:x_end]
        scores = F.resize_images(scores, (ori_H, ori_W))[0].array
        return scores

    def _predict(self, imgs):
        xs = chainer.Variable(self.xp.asarray(imgs))
        with chainer.using_config('train', False):
            scores = F.softmax(self.forward(xs)).array
        return scores

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
        labels = []
        for img in imgs:
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                if self.scales is not None:
                    scores = _multiscale_predict(
                        self._tile_predict, img, self.scales)
                else:
                    scores = self._tile_predict(img)
            labels.append(chainer.backends.cuda.to_cpu(
                self.xp.argmax(scores, axis=0).astype(np.int32)))
        return labels


class PSPNetResNet101(PSPNet):

    """PSPNet with Dilated ResNet101 as the feature extractor.

    .. seealso::
        :class:`chainercv.experimental.links.model.pspnet.PSPNet`

    """

    _extractor_cls = DilatedResNet
    _extractor_kwargs = {'n_layer': 101}
    _extractor_pick = ('res4', 'res5')
    _models = {
        'cityscapes': {
            'param': {'n_class': 19, 'input_size': (713, 713)},
            'url': 'https://chainercv-models.preferred.jp/'
            'pspnet_resnet101_cityscapes_trained_2018_12_19.npz',
        },
        'ade20k': {
            'param': {'n_class': 150, 'input_size': (473, 473)},
            'url': 'https://chainercv-models.preferred.jp/'
            'pspnet_resnet101_ade20k_trained_2018_12_23.npz',
        },
    }


class PSPNetResNet50(PSPNet):

    """PSPNet with Dilated ResNet50 as the feature extractor.

    .. seealso::
        :class:`chainercv.experimental.links.model.pspnet.PSPNet`

    """

    _extractor_cls = DilatedResNet
    _extractor_kwargs = {'n_layer': 50}
    _extractor_pick = ('res4', 'res5')
    _models = {
        'cityscapes': {
            'param': {'n_class': 19, 'input_size': (713, 713)},
            'url': 'https://chainercv-models.preferred.jp/'
            'pspnet_resnet50_cityscapes_trained_2018_12_19.npz',
        },
        'ade20k': {
            'param': {'n_class': 150, 'input_size': (473, 473)},
            'url': 'https://chainercv-models.preferred.jp/'
            'pspnet_resnet50_ade20k_trained_2018_12_23.npz',
        },
    }


def _multiscale_predict(predict_method, img, scales):
    orig_H, orig_W = img.shape[1:]
    scores = []
    orig_img = img
    for scale in scales:
        img = orig_img.copy()
        if scale != 1.0:
            img = transforms.resize(
                img, (int(orig_H * scale), int(orig_W * scale)))
        # This method should return scores
        y = predict_method(img)[None]
        assert y.shape[2:] == img.shape[1:]

        if scale != 1.0:
            y = F.resize_images(y, (orig_H, orig_W)).array
        scores.append(y)
    xp = chainer.backends.cuda.get_array_module(scores[0])
    scores = xp.stack(scores)
    return scores.mean(0)[0]  # (C, H, W)
