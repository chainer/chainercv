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


def _upsample(x):
    return F.unpooling_2d(x, 2, cover_all=False)


class ResidualBlock(chainer.ChainList):
    """ChainList with a residual connection."""

    def __init__(self, *links):
        super(ResidualBlock, self).__init__(*links)

    def forward(self, x):
        h = x
        for link in self:
            h = link(h)
        h += x
        return h


class Darknet53Extractor(chainer.ChainList):
    """A Darknet53 based feature extractor for YOLOv3.

    This is a feature extractor for :class:`~chainercv.links.model.yolo.YOLOv3`
    """

    insize = 416
    grids = (13, 26, 52)

    def __init__(self):
        super(Darknet53Extractor, self).__init__()

        # Darknet53
        self.append(Conv2DBNActiv(32, 3, pad=1, activ=_leaky_relu))
        for k, n_block in enumerate((1, 2, 8, 8, 4)):
            self.append(Conv2DBNActiv(
                32 << (k + 1), 3, stride=2, pad=1, activ=_leaky_relu))
            for _ in range(n_block):
                self.append(ResidualBlock(
                    Conv2DBNActiv(32 << k, 1, activ=_leaky_relu),
                    Conv2DBNActiv(32 << (k + 1), 3, pad=1, activ=_leaky_relu)))

        # additional links
        for i, n in enumerate((512, 256, 128)):
            if i > 0:
                self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu))
            self.append(Conv2DBNActiv(n, 1, activ=_leaky_relu))

    def forward(self, x):
        """Compute feature maps from a batch of images.

        This method extracts feature maps from 3 layers.

        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized to :math:`416\\times 416`.

        Returns:
            list of Variable:
            Each variable contains a feature map.
        """

        ys = []
        h = x
        hs = []
        for i, link in enumerate(self):
            h = link(h)
            if i in {33, 39, 45}:
                ys.append(h)
            elif i in {14, 23}:
                hs.append(h)
            elif i in {34, 40}:
                h = F.concat((_upsample(h), hs.pop()))
        return ys


class YOLOv3(YOLOBase):
    """YOLOv3.

    This is a model of YOLOv3 [#]_.
    This model uses :class:`~chainercv.links.model.yolo.Darknet53Extractor` as
    its feature extractor.

    .. [#] Joseph Redmon, Ali Farhadi.
       YOLOv3: An Incremental Improvement. arXiv 2018.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (string): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the darknet model. \
                The conversion code is \
                `chainercv/examples/yolo/darknet2npz.py`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _models = {
        'voc0712': {
            'param': {'n_fg_class': 20},
            'url': 'https://chainercv-models.preferred.jp/'
            'yolo_v3_voc0712_converted_2018_05_01.npz',
            'cv2': True
        },
    }

    _anchors = (
        ((90, 116), (198, 156), (326, 373)),
        ((61, 30), (45, 62), (119, 59)),
        ((13, 10), (30, 16), (23, 33)))

    def __init__(self, n_fg_class=None, pretrained_model=None):
        super(YOLOv3, self).__init__()

        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        self.n_fg_class = param['n_fg_class']
        self.use_preset('visualize')

        with self.init_scope():
            self.extractor = Darknet53Extractor()
            self.subnet = chainer.ChainList()

        for i, n in enumerate((512, 256, 128)):
            self.subnet.append(chainer.Sequential(
                Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu),
                Convolution2D(
                    len(self._anchors[i]) * (4 + 1 + self.n_fg_class), 1)))

        default_bbox = []
        step = []
        for k, grid in enumerate(self.extractor.grids):
            for v, u in itertools.product(range(grid), repeat=2):
                for h, w in self._anchors[k]:
                    default_bbox.append((v, u, h, w))
                    step.append(self.insize / grid)
        self._default_bbox = np.array(default_bbox, dtype=np.float32)
        self._step = np.array(step, dtype=np.float32)

        if path:
            chainer.serializers.load_npz(path, self, strict=False)

    def to_cpu(self):
        super(YOLOv3, self).to_cpu()
        self._default_bbox = cuda.to_cpu(self._default_bbox)
        self._step = cuda.to_cpu(self._step)

    def to_gpu(self, device=None):
        super(YOLOv3, self).to_gpu(device)
        self._default_bbox = cuda.to_gpu(self._default_bbox, device)
        self._step = cuda.to_gpu(self._step, device)

    def forward(self, x):
        """Compute localization, objectness, and classification from a batch of images.

        This method computes three variables, :obj:`locs`, :obj:`objs`,
        and :obj:`confs`.
        :meth:`self._decode` converts these variables to bounding box
        coordinates and confidence scores.
        These variables are also used in training YOLOv3.

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

        ys = []
        for i, h in enumerate(self.extractor(x)):
            h = self.subnet[i](h)
            h = F.transpose(h, (0, 2, 3, 1))
            h = F.reshape(h, (h.shape[0], -1, 4 + 1 + self.n_fg_class))
            ys.append(h)
        y = F.concat(ys)
        locs = y[:, :, :4]
        objs = y[:, :, 4]
        confs = y[:, :, 5:]
        return locs, objs, confs

    def _decode(self, loc, obj, conf):
        raw_bbox = self._default_bbox.copy()
        raw_bbox[:, :2] += 1 / (1 + self.xp.exp(-loc[:, :2]))
        raw_bbox[:, :2] *= self._step[:, None]
        raw_bbox[:, 2:] *= self.xp.exp(loc[:, 2:])
        raw_bbox[:, :2] -= raw_bbox[:, 2:] / 2
        raw_bbox[:, 2:] += raw_bbox[:, :2]

        obj = 1 / (1 + self.xp.exp(-obj))
        conf = 1 / (1 + self.xp.exp(-conf))
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
