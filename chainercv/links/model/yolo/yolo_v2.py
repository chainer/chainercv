from __future__ import division

import itertools
import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer.links import Convolution2D

from chainercv.links import Conv2DBNActiv
from chainercv import transforms
from chainercv import utils


def _leaky_relu(x):
    return F.leaky_relu(x, slope=0.1)


def _maxpool(x):
    return F.max_pooling_2d(x, 2)


def _reorg(x):
    n, c, h, w = x.shape
    x = F.reshape(x, (n, c // 4, h, 2, w, 2))
    x = F.transpose(x, (0, 3, 5, 1, 2, 4))
    return F.reshape(x, (n, c * 4, h // 2, w // 2))


class Darknet19Extractor(chainer.ChainList):
    """A Darknet19 based feature extractor for YOLOv2.

    This is a feature extractor for :class:`~chainercv.links.model.yolo.YOLOv2`
    """

    insize = 416
    grid = 13

    def __init__(self):
        super().__init__()

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

    def __call__(self, x):
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
                h = _maxpool(h)
        return h


class YOLOv2(chainer.Chain):

    _anchors = (
        (1.73145, 1.3221),
        (4.00944, 3.19275),
        (8.09892, 5.05587),
        (4.84053, 9.47112),
        (10.0071, 11.2364))

    def __init__(self, n_fg_class=None, pretrained_model=None):
        super(YOLOv2, self).__init__()

        self.n_fg_class = n_fg_class
        self.use_preset('visualize')

        with self.init_scope():
            self.extractor = Darknet19Extractor()
            self.subnet = Convolution2D(
                len(self._anchors) * (4 + 1 + self.n_fg_class), 1)

        default_bbox = []
        for v, u in itertools.product(range(self.extractor.grid), repeat=2):
            for h, w in self._anchors:
                default_bbox.append((v, u, h, w))
        self._default_bbox = np.array(default_bbox, dtype=np.float32)

    @property
    def insize(self):
        return self.extractor.insize

    def to_cpu(self):
        super(YOLOv2, self).to_cpu()
        self._default_bbox = cuda.to_cpu(self._default_bbox)
        self._step = cuda.to_cpu(self._step)

    def to_gpu(self, device=None):
        super(YOLOv2, self).to_gpu(device)
        self._default_bbox = cuda.to_gpu(self._default_bbox, device)
        self._step = cuda.to_gpu(self._step, device)

    def __call__(self, x):
        """Compute localization and classification from a batch of images.

        This method computes a variable.
        :func:`self._decode` converts this variable to bounding box
        coordinates and confidence scores.
        This variable is also used in training YOLOv3.

        Args:
            x (chainer.Variable): A variable holding a batch of images.
                The images are preprocessed by :meth:`_prepare`.

        Returns:
            chainer.Variable:
            A variable of float arrays of shape
            :math:`(B, K, 4 + 1 + n\_fg\_class)`,
            where :math:`B` is the number of samples in the batch and
            :math:`K` is the number of default bounding boxes.
        """

        h = self.subnet(self.extractor(x))
        h = F.transpose(h, (0, 2, 3, 1))
        h = F.reshape(h, (h.shape[0], -1, 4 + 1 + self.n_fg_class))
        return h

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`nms_thresh` and
        :obj:`score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'}): A string to determine the
                preset to use.
        """

        if preset == 'visualize':
            self.nms_thresh = 0.45
            self.score_thresh = 0.5
        elif preset == 'evaluate':
            self.nms_thresh = 0.45
            self.score_thresh = 0.005
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _decode(self, loc, conf):
        raw_bbox = self._default_bbox.copy()
        raw_bbox[:, :2] += 1 / (1 + self.xp.exp(-loc[:, :2]))
        raw_bbox[:, :2] *= self.insize / self.extractor.grid
        raw_bbox[:, 2:] *= self.xp.exp(loc[:, 2:])
        raw_bbox[:, 2:] *= self.insize / self.extractor.grid
        raw_bbox[:, :2] -= raw_bbox[:, 2:] / 2
        raw_bbox[:, 2:] += raw_bbox[:, :2]

        raw_score = self.xp.exp(conf[:, 1:])
        raw_score /= raw_score.sum(axis=1, keepdims=True)
        raw_score /= 1 + self.xp.exp(-conf[:, 0, None])

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

            bbox.append(cuda.to_cpu(bbox_l))
            label.append(np.array((l,) * len(bbox_l)))
            score.append(cuda.to_cpu(score_l))

        bbox = np.vstack(bbox).astype(np.float32)
        label = np.hstack(label).astype(np.int32)
        score = np.hstack(score).astype(np.float32)

        return bbox, label, score

    def predict(self, imgs):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """

        x = []
        params = []
        for img in imgs:
            _, H, W = img.shape
            img, param = transforms.resize_contain(
                img / 255, (self.insize, self.insize), fill=0.5,
                return_param=True)
            x.append(self.xp.array(img.astype(np.float32)))
            param['size'] = (H, W)
            params.append(param)

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            y = self(self.xp.stack(x)).array
        locs = y[:, :, :4]
        confs = y[:, :, 4:]

        bboxes = []
        labels = []
        scores = []
        for loc, conf, param in zip(locs, confs, params):
            bbox, label, score = self._decode(loc, conf)

            bbox = transforms.translate_bbox(
                bbox, -self.insize / 2, -self.insize / 2)
            bbox = transforms.resize_bbox(
                bbox, param['scaled_size'], param['size'])
            bbox = transforms.translate_bbox(
                bbox, param['size'][0] / 2, param['size'][1] / 2)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores
