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


def _upsample(x):
    return F.unpooling_2d(x, 2, cover_all=False)


class ResidualBlock(chainer.ChainList):

    def __init__(self, *links):
        super().__init__(*links)

    def __call__(self, x):
        h = x
        for link in self:
            h = link(h)
        h += x
        return h


class Darknet53Extractor(chainer.ChainList):

    insize = 416
    grids = (13, 26, 52)

    def __init__(self):
        super().__init__()

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

    def __call__(self, x):
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


class YOLOv3(chainer.Chain):

    anchors = (
        ((90, 116), (198, 156), (326, 373)),
        ((61, 30), (45, 62), (119, 59)),
        ((13, 10), (30, 16), (23, 33)))

    def __init__(self, n_fg_class=None, pretrained_model=None):
        super().__init__()
        self.n_fg_class = n_fg_class
        self.use_preset('visualize')

        with self.init_scope():
            self.extractor = Darknet53Extractor()
            self.subnet = chainer.ChainList()

        for n in (512, 256, 128):
            self.subnet.append(chainer.Sequential(
                Conv2DBNActiv(n * 2, 3, pad=1, activ=_leaky_relu),
                Convolution2D(3 * (4 + 1 + self.n_fg_class), 1)))

        default_bbox = []
        step = []
        for k, grid in enumerate(self.extractor.grids):
            for v, u in itertools.product(range(grid), repeat=2):
                for h, w in self.anchors[k]:
                    default_bbox.append((v, u, h, w))
                    step.append(self.insize / grid)
        self._default_bbox = np.array(default_bbox, dtype=np.float32)
        self._step = np.array(step, dtype=np.float32)

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self, strict=False)

    @property
    def insize(self):
        return self.extractor.insize

    def to_cpu(self):
        super().to_cpu()
        self._default_bbox = cuda.to_cpu(self._default_bbox)
        self._step = cuda.to_cpu(self._step)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self._default_bbox = cuda.to_gpu(self._default_bbox, device)
        self._step = cuda.to_gpu(self._step, device)

    def __call__(self, x):
        ys = []
        for i, h in enumerate(self.extractor(x)):
            h = self.subnet[i](h)
            h = F.transpose(h, (0, 2, 3, 1))
            h = F.reshape(h, (h.shape[0], -1, 4 + 1 + self.n_fg_class))
            ys.append(h)
        return F.concat(ys)

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.45
            self.score_thresh = 0.6
        elif preset == 'evaluate':
            self.nms_thresh = 0.45
            self.score_thresh = 0.01
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs):
        x = []
        params = []
        for img in imgs:
            _, H, W = img.shape
            img, param = transforms.resize_contain(
                img / 255, (self.insize, self.insize), fill=0.5,
                return_param=True)
            x.append(self.xp.array(img))
            param['size'] = (H, W)
            params.append(param)

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            y = self(self.xp.stack(x)).array
        locs = y[:, :, :4]
        objs = y[:, :, 4]
        confs = y[:, :, 5:]

        bboxes = []
        labels = []
        scores = []
        for loc, obj, conf, param in zip(locs, objs, confs, params):
            bbox = self._default_bbox.copy()
            bbox[:, :2] += 1 / (1 + self.xp.exp(-loc[:, :2]))
            bbox[:, :2] *= self._step[:, None]
            bbox[:, 2:] *= self.xp.exp(loc[:, 2:])
            bbox[:, :2] -= bbox[:, 2:] / 2
            bbox[:, 2:] += bbox[:, :2]

            bbox = transforms.translate_bbox(
                bbox, -self.insize / 2, -self.insize / 2)
            bbox = transforms.resize_bbox(
                bbox, param['scaled_size'], param['size'])
            bbox = transforms.translate_bbox(
                bbox, param['size'][0] / 2, param['size'][1] / 2)

            score = 1 / (1 + self.xp.exp(-obj))

            conf = self.xp.exp(conf)
            conf /= conf.sum(axis=1, keepdims=True)
            label = conf.argmax(axis=1)

            mask = score >= self.score_thresh
            bbox = bbox[mask]
            label = label[mask]
            score = score[mask]

            indices = utils.non_maximum_suppression(
                bbox, self.nms_thresh, score)
            bbox = bbox[indices]
            label = label[indices]
            score = score[indices]

            bboxes.append(cuda.to_cpu(bbox))
            labels.append(cuda.to_cpu(label))
            scores.append(cuda.to_cpu(score))

        return bboxes, labels, scores
