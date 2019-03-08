from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links.connection import Conv2DBNActiv
from chainercv.links.connection import SeparableConv2DBNActiv
from chainercv.links.model.deeplab.aspp import SeparableASPP
from chainercv.links.model.deeplab.xception import Xception65
from chainercv.transforms import resize
from chainercv import utils

import numpy as np


class Decoder(chainer.Chain):

    """Decoder for DeepLab V3+.

    Args:
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        proj_channels (int): Number of channels of output of
            first 1x1 convolution.
        depth_channels (int): Number of channels of output of
            convolution after concatenation.
        bn_kwargs (dict): Keywod arguments passed to initialize the batch
            normalization layers of :class:`chainercv.links.Conv2DBNActiv` and
            :class:`chainercv.links.SeparableConv2DBNActiv`.

    """

    def __init__(self, in_channels, out_channels, proj_channels,
                 depth_channels, bn_kwargs={}):
        super(Decoder, self).__init__()

        with self.init_scope():
            self.feature_proj = Conv2DBNActiv(in_channels, proj_channels, 1)
            concat_channels = in_channels+proj_channels
            self.conv1 = SeparableConv2DBNActiv(
                concat_channels, depth_channels, 3, 1, 1, 1, nobias=True,
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.conv2 = SeparableConv2DBNActiv(
                depth_channels, depth_channels, 3, 1, 1, 1, nobias=True,
                dw_activ=F.relu, pw_activ=F.relu, bn_kwargs=bn_kwargs)
            self.conv_logits = L.Convolution2D(
                depth_channels, out_channels, 1, 1, 0)

    def __call__(self, x, pool):
        x = self.feature_proj(x)
        pool = F.resize_images(pool, x.shape[2:])
        h = F.concat((pool, x), axis=1)
        h = self.conv1(h)
        h = self.conv2(h)
        logits = self.conv_logits(h)

        return logits


class DeepLabV3plus(chainer.Chain):

    """Base class of DeepLab V3+.

    Args:
        fature_extractor (callable): Feature extractor network.
            This network should return lowlevel and highlevel feature maps
            as :obj:`(lowlevel, highlevel)`.
        aspp (callable): ASPP network.
        decoder (callable): Decoder network.
        min_input_size (int or tuple of ints): Minimum image size of inputs.
            if height or width is lower than this values, input images are
            padded to be this shape. The default value is :obj:`(513, 513)`
        scales (tuple of floats): Scales for multi-scale prediction.
            Final outputs are averaged after softmax activation.
            The default value is :obj:`(1.0,)`.
        flip (bool): When this is true, a left-right flipped images are
            also input and finally averaged. When :obj:`len(scales)` are
            more than 1, flipped prediction is performed in each scales.
            The default value is :obj:`False`

    """

    def __init__(self, feature_extractor, aspp, decoder,
                 min_input_size, scales=(1.0,), flip=False):
        super(DeepLabV3plus, self).__init__()

        if not isinstance(min_input_size, (list, tuple)):
            min_input_size = (int(min_input_size), int(min_input_size))
        self.min_input_size = min_input_size
        self.scales = scales
        self.flip = flip

        with self.init_scope():
            self.feature_extractor = feature_extractor
            self.aspp = aspp
            self.decoder = decoder

    def prepare(self, image):
        """Preprocess an image for feature extraction.

        1. padded by mean pixel defined in feature extractor.
        2. scaled to [-1.0, 1.0]

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            image (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """

        _, H, W = image.shape

        # Pad image and label to have dimensions >= min_input_size
        h = max(self.min_input_size[0], H)
        w = max(self.min_input_size[1], W)

        # Pad image with mean pixel value.
        mean = self.feature_extractor.mean
        bg = np.zeros((3, h, w), dtype=np.float32) + mean
        bg[:, :H, :W] = image
        image = bg

        # scale to [-1.0, 1.0]
        image = image / 127.5 - 1.0

        return image

    def __call__(self, x):
        lowlevel, highlevel = self.feature_extractor(x)
        highlevel = self.aspp(highlevel)
        h = self.decoder(lowlevel, highlevel)
        return h

    def _get_proba(self, img, scale, flip):
        if flip:
            img = img[:, :, ::-1]

        _, H, W = img.shape
        if scale == 1.0:
            h, w = H, W
        else:
            h, w = int(H * scale), int(W * scale)
            img = resize(img, (h, w))

        img = self.prepare(img)

        x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
        x = self.__call__(x)
        x = F.softmax(x, axis=1)
        score = F.resize_images(x, img.shape[1:])[0, :, :h, :w].array
        score = chainer.backends.cuda.to_cpu(score)

        if scale != 1.0:
            score = resize(score, (H, W))

        if flip:
            score = score[:, :, ::-1]

        return score

    def predict(self, imgs):
        """Conduct semantic segmentation from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.

        Returns:
            list of numpy.ndarray:

            List of integer labels predicted from each image in the input list.

        """

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            labels = []
            score = 0
            n_aug = len(self.scales) if self.flip else len(self.scales) * 2

            for img in imgs:
                for scale in self.scales:
                    score += self._get_proba(img, scale, False) / n_aug
                    if self.flip:
                        score += self._get_proba(img, scale, True) / n_aug

                label = np.argmax(score, axis=0).astype(np.int32)
                labels.append(label)
        return labels


class DeepLabV3plusXception65(DeepLabV3plus):
    _models = {
        'voc': {
            'param': {
                'n_class': 21,
                'min_input_size': (513, 513),
                'scales': (1.0,),
                'flip': False,
                'extractor_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-3},
                },
                'aspp_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-5},
                },
                'decoder_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-5},
                },
            },
            'overwritable': ('scales', 'flip'),
            'url': 'https://chainercv-models.preferred.jp/'
            'deeplabv3plus_xception65_voc_converted_2019_02_15.npz',
        },
        'cityscapes': {
            'param': {
                'n_class': 19,
                'min_input_size': (1025, 2049),
                'scales': (1.0,),
                'flip': False,
                'extractor_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-3},
                },
                'aspp_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-5},
                },
                'decoder_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-5},
                },
            },
            'overwritable': ('scales', 'flip'),
            'url': 'https://chainercv-models.preferred.jp/'
            'deeplabv3plus_xception65_cityscapes_converted_2019_02_15.npz',
        },
        'ade20k': {
            'param': {
                'n_class': 150,
                'min_input_size': (513, 513),
                'scales': (1.0,),
                'flip': False,
                'extractor_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-3},
                },
                'aspp_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-5},
                },
                'decoder_kwargs': {
                    'bn_kwargs': {'decay': 0.9997, 'eps': 1e-5},
                },
            },
            'overwritable': ('scales', 'flip'),
            'url': 'https://chainercv-models.preferred.jp/'
            'deeplabv3plus_xception65_ade20k_converted_2019_03_08.npz',
        }
    }

    def __init__(self, n_class=None, pretrained_model=None,
                 min_input_size=None, scales=None, flip=None,
                 extractor_kwargs=None, aspp_kwargs=None, decoder_kwargs=None):
        param, path = utils.prepare_pretrained_model(
            {'n_class': n_class, 'min_input_size': min_input_size,
             'scales': scales, 'flip': flip,
             'extractor_kwargs': extractor_kwargs,
             'aspp_kwargs': aspp_kwargs, 'decoder_kwargs': decoder_kwargs},
            pretrained_model, self._models,
            default={'min_input_size': (513, 513)})

        super(DeepLabV3plusXception65, self).__init__(
            Xception65(**param['extractor_kwargs']),
            SeparableASPP(2048, 256, **param['aspp_kwargs']),
            Decoder(256, param['n_class'], 48, 256, **param['decoder_kwargs']),
            min_input_size=param['min_input_size'], scales=param['scales'],
            flip=param['flip'])

        if path:
            chainer.serializers.load_npz(path, self)
