from __future__ import division

import warnings
from math import ceil

import numpy as np
import six

import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.utils import download_model

try:
    from chainermn.links import MultiNodeBatchNormalization
except Exception:
    warnings.warn('To perform batch normalization with multiple GPUs or '
                  'multiple nodes, MultiNodeBatchNormalization link is '
                  'needed. Please install ChainerMN: '
                  'pip install pip install git+git://github.com/chainer/'
                  'chainermn.git@distributed-batch-normalization')


class ConvBNReLU(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        comm = chainer.config.comm
        w = chainer.config.initialW
        with self.init_scope():
            if dilation > 1:
                self.conv = L.DilatedConvolution2D(
                    in_ch, out_ch, ksize, stride, pad, dilation, True, w)
            else:
                self.conv = L.Convolution2D(
                    in_ch, out_ch, ksize, stride, pad, True, w)
            if comm is not None:
                self.bn = MultiNodeBatchNormalization(
                    out_ch, comm, eps=1e-5, decay=0.95)
            else:
                self.bn = L.BatchNormalization(out_ch, eps=1e-5, decay=0.95)

    def __call__(self, x, relu=True):
        h = self.bn(self.conv(x))
        return h if not relu else F.relu(h)


class PyramidPoolingModule(chainer.ChainList):

    def __init__(self, in_ch, feat_size, pyramids):
        super(PyramidPoolingModule, self).__init__(
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0),
            ConvBNReLU(in_ch, in_ch // len(pyramids), 1, 1, 0))
        if isinstance(feat_size, int):
            self.ksizes = (feat_size // np.array(pyramids)).tolist()
        elif isinstance(feat_size, (list, tuple)) and len(feat_size) == 2:
            kh = (feat_size[0] // np.array(pyramids)).tolist()
            kw = (feat_size[1] // np.array(pyramids)).tolist()
            self.ksizes = list(zip(kh, kw))

    def __call__(self, x):
        ys = [x]
        h, w = x.shape[2:]
        for f, ksize in zip(self, self.ksizes):
            y = F.average_pooling_2d(x, ksize, ksize)  # Pad should be 0!
            y = f(y)  # Reduce num of channels
            y = F.resize_images(y, (h, w))
            ys.append(y)
        return F.concat(ys, axis=1)


class BottleneckConv(chainer.Chain):

    def __init__(self, in_ch, mid_ch, out_ch, stride=2, dilate=False):
        mid_stride = chainer.config.mid_stride
        super(BottleneckConv, self).__init__()
        with self.init_scope():
            self.cbr1 = ConvBNReLU(
                in_ch, mid_ch, 1, 1 if mid_stride else stride, 0)
            if dilate:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, dilate, dilate)
            else:
                self.cbr2 = ConvBNReLU(
                    mid_ch, mid_ch, 3, stride if mid_stride else 1, 1)
            self.cbr3 = ConvBNReLU(mid_ch, out_ch, 1, 1, 0)
            self.cbr4 = ConvBNReLU(in_ch, out_ch, 1, stride, 0)

    def __call__(self, x):
        h = self.cbr1(x)
        h = self.cbr2(h)
        h1 = self.cbr3(h, relu=False)
        h2 = self.cbr4(x, relu=False)
        return F.relu(h1 + h2)


class BottleneckIdentity(chainer.Chain):

    def __init__(self, in_ch, mid_ch, dilate=False):
        super(BottleneckIdentity, self).__init__()
        with self.init_scope():
            self.cbr1 = ConvBNReLU(in_ch, mid_ch, 1, 1, 0)
            if dilate:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, dilate, dilate)
            else:
                self.cbr2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
            self.cbr3 = ConvBNReLU(mid_ch, in_ch, 1, 1, 0)

    def __call__(self, x):
        h = self.cbr1(x)
        h = self.cbr2(h)
        h = self.cbr3(h, relu=False)
        return F.relu(h + x)


class ResBlock(chainer.ChainList):

    def __init__(self, n_layer, in_ch, mid_ch, out_ch, stride):
        super(ResBlock, self).__init__()
        self.add_link(BottleneckConv(in_ch, mid_ch, out_ch, stride))
        for _ in six.moves.xrange(1, n_layer):
            self.add_link(BottleneckIdentity(out_ch, mid_ch))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class DilatedResBlock(chainer.ChainList):

    def __init__(self, n_layer, in_ch, mid_ch, out_ch, dilate):
        super(DilatedResBlock, self).__init__()
        self.add_link(BottleneckConv(in_ch, mid_ch, out_ch, 1, dilate))
        for _ in six.moves.xrange(1, n_layer):
            self.add_link(BottleneckIdentity(out_ch, mid_ch, dilate))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class DilatedFCN(chainer.Chain):

    def __init__(self, n_blocks):
        super(DilatedFCN, self).__init__()
        with self.init_scope():
            self.cbr1_1 = ConvBNReLU(None, 64, 3, 2, 1)
            self.cbr1_2 = ConvBNReLU(64, 64, 3, 1, 1)
            self.cbr1_3 = ConvBNReLU(64, 128, 3, 1, 1)
            self.res2 = ResBlock(n_blocks[0], 128, 64, 256, 1)
            self.res3 = ResBlock(n_blocks[1], 256, 128, 512, 2)
            self.res4 = DilatedResBlock(n_blocks[2], 512, 256, 1024, 2)
            self.res5 = DilatedResBlock(n_blocks[3], 1024, 512, 2048, 4)

    def __call__(self, x):
        h = self.cbr1_3(self.cbr1_2(self.cbr1_1(x)))  # 1/2
        h = F.max_pooling_2d(h, 3, 2, 1)  # 1/4
        h = self.res2(h)
        h = self.res3(h)  # 1/8
        if chainer.config.train:
            h1 = self.res4(h)
            h2 = self.res5(h1)
            return h1, h2
        else:
            h = self.res4(h)
            return self.res5(h)


class PSPNet(chainer.Chain):

    """Pyramid Scene Parsing Network

    This Chain supports any depth of ResNet and any pyramid levels for
    the pyramid pooling module (PPM).

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`voc2012`: Loads weights trained with the trainval split of \
        PASCAL VOC2012 Semantic Segmentation Dataset.
    * :obj:`cityscapes`: Loads weights trained with Cityscapes dataset.
    * :obj:`ade20k`: Loads weights trained with ADE20K dataset.

    Args:
        n_class (int): The number of channels in the last convolution layer.
        input_size (int or iterable of ints): The input image size. If a
            single integer is given, it's treated in the same way as if
            a tuple of (input_size, input_size) is given. If an iterable object
            is given, it should mean (height, width) of the input images.
        n_blocks (list of int): Numbers of layers in ResNet. Typically,
            [3, 4, 23, 3] for ResNet101 (used for PASCAL VOC2012 and
            Cityscapes in the original paper) and [3, 4, 6, 3] for ResNet50
            (used for ADE20K datset in the original paper).
        pyramids (list of int): The number of division to the feature map in
            each pyramid level. The length of this list will be the number of
            levels of pyramid in the pyramid pooling module. In each pyramid,
            an average pooling is applied to the feature map with the kernel
            size of the corresponding value in this list.
        mid_stride (bool): If True, spatial dimention reduction in bottleneck
            modules in ResNet part will be done at the middle 3x3 convolution.
            It means that the stride of the middle 3x3 convolution will be two.
            Otherwise (if it's set to False), the stride of the first 1x1
            convolution in the bottleneck module will be two as in the original
            ResNet and Deeplab v2.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :meth:`prepare`.
        comm (chainermn.communicator or None): If a ChainerMN communicator is
            given, it will be used for distributed batch normalization during
            training. If None, all batch normalization links will not share
            the input vectors among GPUs before calculating mean and variance.
            The original PSPNet implementation uses distributed batch
            normalization.
        pretrained_model (str): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.

    """

    _models = {
        'voc2012': {
            'n_class': 21,
            'input_size': (473, 473),
            'n_blocks': [3, 4, 23, 3],
            'feat_size': 60,
            'mid_stride': True,
            'pyramids': [6, 3, 2, 1],
            'mean': np.array([123.68, 116.779, 103.939]),
            'url': 'https://github.com/mitmul/chainer-pspnet/releases/download'
                   '/ChainerCV_PSPNet/pspnet101_VOC2012_473_reference.npz'
        },
        'cityscapes': {
            'n_class': 19,
            'input_size': (713, 713),
            'n_blocks': [3, 4, 23, 3],
            'feat_size': 90,
            'mid_stride': True,
            'pyramids': [6, 3, 2, 1],
            'mean': np.array([123.68, 116.779, 103.939]),
            'url': 'https://github.com/mitmul/chainer-pspnet/releases/download'
                   '/ChainerCV_PSPNet/pspnet101_cityscapes_713_reference.npz'
        },
        'ade20k': {
            'n_class': 150,
            'input_size': (473, 473),
            'n_blocks': [3, 4, 6, 3],
            'feat_size': 60,
            'mid_stride': True,
            'pyramids': [6, 3, 2, 1],
            'mean': np.array([123.68, 116.779, 103.939]),
            'url': 'https://github.com/mitmul/chainer-pspnet/releases/download'
                   '/ChainerCV_PSPNet/pspnet50_ADE20K_473_reference.npz'
        }
    }

    def __init__(self, n_class=None, input_size=None, n_blocks=None,
                 pyramids=None, mid_stride=None, mean=None, comm=None,
                 pretrained_model=None, initialW=None):
        super(PSPNet, self).__init__()

        if pretrained_model in self._models:
            if 'n_class' in self._models[pretrained_model]:
                n_class = self._models[pretrained_model]['n_class']
            if 'input_size' in self._models[pretrained_model]:
                input_size = self._models[pretrained_model]['input_size']
            if 'n_blocks' in self._models[pretrained_model]:
                n_blocks = self._models[pretrained_model]['n_blocks']
            if 'pyramids' in self._models[pretrained_model]:
                pyramids = self._models[pretrained_model]['pyramids']
            if 'mid_stride' in self._models[pretrained_model]:
                mid_stride = self._models[pretrained_model]['mid_stride']
            if 'mean' in self._models[pretrained_model]:
                mean = self._models[pretrained_model]['mean']
                self._use_pretrained_model = True

        chainer.config.mid_stride = mid_stride
        chainer.config.comm = comm

        if initialW is None:
            chainer.config.initialW = chainer.initializers.HeNormal()
        else:
            chainer.config.initialW = initialW

        if not isinstance(input_size, (list, tuple)):
            input_size = (int(input_size), int(input_size))

        with self.init_scope():
            self.input_size = input_size
            self.trunk = DilatedFCN(n_blocks=n_blocks)

            # To calculate auxirally loss
            if chainer.config.train:
                self.cbr_aux = ConvBNReLU(None, 512, 3, 1, 1)
                self.out_aux = L.Convolution2D(
                    512, n_class, 3, 1, 1, False, initialW)

            # Main branch
            feat_size = (input_size[0] // 8, input_size[1] // 8)
            self.ppm = PyramidPoolingModule(2048, feat_size, pyramids)
            self.cbr_main = ConvBNReLU(4096, 512, 3, 1, 1)
            self.out_main = L.Convolution2D(
                512, n_class, 1, 1, 0, False, initialW)

        self.mean = mean

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
            self._use_pretrained_model = True
            print('Pre-trained model has been loaded:', pretrained_model)
        elif pretrained_model:
            self._use_pretrained_model = False
            chainer.serializers.load_npz(pretrained_model, self)
            print('Pre-trained model has been loaded:', pretrained_model)
        else:
            self._use_pretrained_model = False

    @property
    def n_class(self):
        return self.out_main.out_channels

    def __call__(self, x):
        """Forward computation of PSPNet

        Args:
            x: Input array or Variable.

        Returns:
            Training time: it returns the outputs from auxiliary branch and the
                main branch. So the returned value is a tuple of two Variables.
            Inference time: it returns the output of the main branch. So the
                returned value is a sinle Variable which forms
                ``(N, n_class, H, W)`` where ``N`` is the batchsize and
                ``n_class`` is the number of classes specified in the
                constructor. ``H, W`` is the input image size.

        """
        if chainer.config.train:
            aux, h = self.trunk(x)
            aux = F.dropout(self.cbr_aux(aux), ratio=0.1)
            aux = self.out_aux(aux)
            aux = F.resize_images(aux, x.shape[2:])
        else:
            h = self.trunk(x)

        h = self.ppm(h)
        h = F.dropout(self.cbr_main(h), ratio=0.1)
        h = self.out_main(h)
        h = F.resize_images(h, x.shape[2:])

        if chainer.config.train:
            return aux, h
        else:
            return h

    def prepare(self, img):
        """Preprocess an image for feature extraction.

        The image is subtracted by a mean image value :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        if self.mean is not None:
            img -= self.mean[:, None, None]
            img = img.astype(np.float32, copy=False)
            if self._use_pretrained_model:
                # Pre-trained model is trained for BGR images
                img = img[::-1, ...]
        return img

    def _predict(self, img):
        img = chainer.Variable(self.xp.asarray(img))
        with chainer.using_config('train', False):
            score = self.__call__(img)
        return chainer.cuda.to_cpu(F.softmax(score).data)

    def _pad_img(self, img):
        if img.shape[1] < self.input_size[0]:
            pad_h = self.input_size[0] - img.shape[1]
            img = np.pad(img, ((0, 0), (0, pad_h), (0, 0)), 'constant')
        else:
            pad_h = 0
        if img.shape[2] < self.input_size[1]:
            pad_w = self.input_size[1] - img.shape[2]
            img = np.pad(img, ((0, 0), (0, 0), (0, pad_w)), 'constant')
        else:
            pad_w = 0
        return img, pad_h, pad_w

    def _tile_predict(self, img):
        ori_rows, ori_cols = img.shape[1:]
        long_size = max(ori_rows, ori_cols)

        # When padding input patches is needed
        if long_size > max(self.input_size):
            count = np.zeros((ori_rows, ori_cols))
            pred = np.zeros((1, self.n_class, ori_rows, ori_cols))
            stride_rate = 2 / 3.
            stride = (ceil(self.input_size[0] * stride_rate),
                      ceil(self.input_size[1] * stride_rate))
            hh = ceil((ori_rows - self.input_size[0]) / stride[0]) + 1
            ww = ceil((ori_cols - self.input_size[1]) / stride[1]) + 1
            for yy in six.moves.xrange(hh):
                for xx in six.moves.xrange(ww):
                    sy, sx = yy * stride[0], xx * stride[1]
                    ey, ex = sy + self.input_size[0], sx + self.input_size[1]
                    img_sub = img[:, sy:ey, sx:ex]
                    img_sub, pad_h, pad_w = self._pad_img(img_sub)

                    # Take average of pred and pred from flipped image
                    psub1 = self._predict(img_sub[np.newaxis])
                    psub2 = self._predict(img_sub[np.newaxis, :, :, ::-1])
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.

                    if sy + self.input_size[0] > ori_rows:
                        psub = psub[:, :, :-pad_h, :]
                    if sx + self.input_size[1] > ori_cols:
                        psub = psub[:, :, :, :-pad_w]
                    pred[:, :, sy:ey, sx:ex] = psub
                    count[sy:ey, sx:ex] += 1
            score = (pred / count[None, None, ...]).astype(np.float32)
        else:
            img, pad_h, pad_w = self._pad_img(img)
            pred1 = self._predict(img[np.newaxis])
            pred2 = self._predict(img[np.newaxis, :, :, ::-1])
            pred = (pred1 + pred2[:, :, :, ::-1]) / 2.
            score = pred[
                :, :, :self.input_size[0] - pad_h, :self.input_size[1] - pad_w]
        score = F.resize_images(score, (ori_rows, ori_cols))[0].data
        return score / score.sum(axis=0)

    def predict(self, imgs, argmax=True):
        """Conduct semantic segmentation from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images should be in CHW order.
            argmax (bool): Whether it performs argmax to the output label
                predictions over the channel axis or not. The default is True.

        Returns:
            list of numpy.ndarray: List of predictions from each image in the
                input list. Note that if you specified ``argmax=True``, each
                prediction is resulting integer label and the number of
                dimensions is two (:math:`(H, W)`). Otherwise, the output will
                be a probability map calculated by the model and its number of
                dimensions will be three (:math:`(C, H, W)`).

        """
        labels = []
        for img in imgs:
            with chainer.using_config('train', False):
                x = self.prepare(img)
                score = self._tile_predict(x)
            label = chainer.cuda.to_cpu(score)
            if argmax:
                label = np.argmax(score, axis=0).astype(np.int32)
            labels.append(label)
        return labels
