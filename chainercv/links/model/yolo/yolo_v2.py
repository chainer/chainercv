from __future__ import division

import itertools
import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer.links import Convolution2D

from chainercv.links import Conv2DBNActiv
from chainercv import utils

from chainercv.links.model.yolo.yolo_base import YOLOBase


def _leaky_relu(x):
    return F.leaky_relu(x, slope=0.1)


def _maxpool(x, ksize, stride=None):
    if stride is None:
        stride = ksize

    h = F.max_pooling_2d(x, ksize, stride=stride, pad=ksize - stride)
    if ksize > stride:
        h = h[:, :, ksize - stride:, ksize - stride:]
    return h


def _reorg(x):
    n, c, h, w = x.shape
    x = F.reshape(x, (n, c // 4, h, 2, w, 2))
    x = F.transpose(x, (0, 3, 5, 1, 2, 4))
    return F.reshape(x, (n, c * 4, h // 2, w // 2))


class YOLOv2Base(YOLOBase):
    """Base class for YOLOv2 and YOLOv2Tiny.

    A subclass of this class should have :obj:`_extractor`,
    :obj:`_models`, and :obj:`_anchors`.
    """

    def __init__(self, n_fg_class=None, pretrained_model=None):
        super(YOLOv2Base, self).__init__()

        param, path = utils.prepare_model_param(locals(), self._models)

        self.n_fg_class = param['n_fg_class']
        self.use_preset('visualize')

        with self.init_scope():
            self.extractor = self._extractor()
            self.subnet = Convolution2D(
                len(self._anchors) * (4 + 1 + self.n_fg_class), 1)

        default_bbox = []
        for v, u in itertools.product(range(self.extractor.grid), repeat=2):
            for h, w in self._anchors:
                default_bbox.append((v, u, h, w))
        self._default_bbox = np.array(default_bbox, dtype=np.float32)

        if path:
            chainer.serializers.load_npz(path, self, strict=False)

    def to_cpu(self):
        super(YOLOv2Base, self).to_cpu()
        self._default_bbox = cuda.to_cpu(self._default_bbox)

    def to_gpu(self, device=None):
        super(YOLOv2Base, self).to_gpu(device)
        self._default_bbox = cuda.to_gpu(self._default_bbox, device)

    def forward(self, x):
        """Compute localization, objectness, and classification from a batch of images.

        This method computes three variables, :obj:`locs`, :obj:`objs`,
        and :obj:`confs`.
        :meth:`self._decode` converts these variables to bounding box
        coordinates and confidence scores.
        These variables are also used in training YOLOv2.

        Args:
            x (chainer.Variable): A variable holding a batch of images.

        Returns:
            tuple of chainer.Variable:
            This method returns three variables, :obj:`locs`,
            :obj:`objs`, and :obj:`confs`.

            * **locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **objs**: A variable of float arrays of shape \
                :math:`(B, K)`.
            * **confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class)`.
        """

        h = self.subnet(self.extractor(x))
        h = F.transpose(h, (0, 2, 3, 1))
        h = F.reshape(h, (h.shape[0], -1, 4 + 1 + self.n_fg_class))
        locs = h[:, :, :4]
        objs = h[:, :, 4]
        confs = h[:, :, 5:]
        return locs, objs, confs

    def _decode(self, loc, obj, conf):
        raw_bbox = self._default_bbox.copy()
        raw_bbox[:, :2] += 1 / (1 + self.xp.exp(-loc[:, :2]))
        raw_bbox[:, 2:] *= self.xp.exp(loc[:, 2:])
        raw_bbox[:, :2] -= raw_bbox[:, 2:] / 2
        raw_bbox[:, 2:] += raw_bbox[:, :2]
        raw_bbox *= self.insize / self.extractor.grid

        obj = 1 / (1 + self.xp.exp(-obj))
        conf = self.xp.exp(conf)
        conf /= conf.sum(axis=1, keepdims=True)
        raw_score = obj[:, None] * conf

        bbox = []
        label = []
        score = []
        for l in range(self.n_fg_class):
            bbox_l = raw_bbox
            score_l = raw_score[:, l]

            mask = score_l >= self.score_thresh
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            indices = utils.non_maximum_suppression(
                bbox_l, self.nms_thresh, score_l)
            bbox_l = bbox_l[indices]
            score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append(self.xp.array((l,) * len(bbox_l)))
            score.append(score_l)

        bbox = self.xp.vstack(bbox).astype(np.float32)
        label = self.xp.hstack(label).astype(np.int32)
        score = self.xp.hstack(score).astype(np.float32)

        return bbox, label, score


class Darknet19Extractor(chainer.ChainList):
    """A Darknet19 based feature extractor for YOLOv2.

    This is a feature extractor for :class:`~chainercv.links.model.yolo.YOLOv2`
    """

    insize = 416
    grid = 13

    def __init__(self):
        super(Darknet19Extractor, self).__init__()

        # Darknet19
        for k, n_conv in enumerate((1, 1, 3, 3, 5, 5)):
            for i in range(n_conv):
                if i % 2 == 0:
                    self.append(
                        Conv2DBNActiv(32 << k, 3, pad=1, activ=_leaky_relu))
                else:
                    self.append(
                        Conv2DBNActiv(32 << (k - 1), 1, activ=_leaky_relu))

        # additional links
        self.append(Conv2DBNActiv(1024, 3, pad=1, activ=_leaky_relu))
        self.append(Conv2DBNActiv(1024, 3, pad=1, activ=_leaky_relu))
        self.append(Conv2DBNActiv(64, 1, activ=_leaky_relu))
        self.append(Conv2DBNActiv(1024, 3, pad=1, activ=_leaky_relu))

    def forward(self, x):
        """Compute a feature map from a batch of images.

        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized to :math:`416\\times 416`.

        Returns:
            Variable:
        """

        h = x
        for i, link in enumerate(self):
            h = link(h)
            if i == 12:
                tmp = h
            elif i == 19:
                h, tmp = tmp, h
            elif i == 20:
                h = F.concat((_reorg(h), tmp))
            if i in {0, 1, 4, 7, 12}:
                h = _maxpool(h, 2)
        return h


class YOLOv2(YOLOv2Base):
    """YOLOv2.

    This is a model of YOLOv2 [#]_.
    This model uses :class:`~chainercv.links.model.yolo.Darknet19Extractor` as
    its feature extractor.

    .. [#] Joseph Redmon, Ali Farhadi.
       YOLO9000: Better, Faster, Stronger. CVPR 2017.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (string): The weight file to be loaded.
            This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
            The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the darknet model \
                provided by `the original implementation \
                <https://pjreddie.com/darknet/yolov2/>`_. \
                The conversion code is \
                `chainercv/examples/yolo/darknet2npz.py`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    preset_params = {
        'voc': {'n_fg_class': 20},
    }
    _extractor = Darknet19Extractor

    _models = {
        'voc0712': {
            'param': preset_params['voc'],
            'url': 'https://chainercv-models.preferred.jp/'
            'yolo_v2_voc0712_converted_2018_05_03.npz',
            'cv2': True
        },
    }

    _anchors = (
        (1.73145, 1.3221),
        (4.00944, 3.19275),
        (8.09892, 5.05587),
        (4.84053, 9.47112),
        (10.0071, 11.2364))
