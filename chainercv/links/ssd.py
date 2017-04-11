from __future__ import division

import itertools
import numpy as np
import os

import chainer
from chainer.dataset import download
import chainer.functions as F
import chainer.links as L
import chainer.links.caffe.caffe_function as caffe
from chainer import serializers

from chainercv import transforms


class _Normalize(chainer.Link):

    def __init__(self, n_channels, eps=1e-6):
        super(_Normalize, self).__init__()
        self.eps = eps
        self.add_param('scale', n_channels)

    def __call__(self, x):
        norm = F.sqrt(F.sum(F.square(x), axis=1) + self.eps)
        norm = F.broadcast_to(norm[:, np.newaxis], x.shape)
        scale = F.broadcast_to(self.scale[:, np.newaxis, np.newaxis], x.shape)
        return x * scale / norm


class _CaffeFunction(caffe.CaffeFunction):

    def __init__(self, model_path):
        super(_CaffeFunction, self).__init__(model_path)

    @caffe._layer('Normalize', None)
    def _setup_normarize(self, layer):
        blobs = layer.blobs
        func = _Normalize(caffe._get_num(blobs[0]))
        func.scale.data[:] = np.array(blobs[0].data)
        self.add_link(layer.name, func)

    @caffe._layer('AnnotatedData', None)
    @caffe._layer('Flatten', None)
    @caffe._layer('MultiBoxLoss', None)
    @caffe._layer('Permute', None)
    @caffe._layer('PriorBox', None)
    def _skip_layer(self, _):
        pass


class SSD300(chainer.Chain):

    mean = (104, 117, 123)
    n_classes = 20
    insize = 300
    grids = (38, 19, 10, 5, 3, 1)
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)]
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    variance = (0.1, 0.2)

    def __init__(self, pretrained_model='auto'):
        super(SSD300, self).__init__(
            conv1_1=L.Convolution2D(None, 64, 3, pad=1),
            conv1_2=L.Convolution2D(None, 64, 3, pad=1),

            conv2_1=L.Convolution2D(None, 128, 3, pad=1),
            conv2_2=L.Convolution2D(None, 128, 3, pad=1),

            conv3_1=L.Convolution2D(None, 256, 3, pad=1),
            conv3_2=L.Convolution2D(None, 256, 3, pad=1),
            conv3_3=L.Convolution2D(None, 256, 3, pad=1),

            conv4_1=L.Convolution2D(None, 512, 3, pad=1),
            conv4_2=L.Convolution2D(None, 512, 3, pad=1),
            conv4_3=L.Convolution2D(None, 512, 3, pad=1),
            norm4=_Normalize(512),

            conv5_1=L.DilatedConvolution2D(None, 512, 3, pad=1),
            conv5_2=L.DilatedConvolution2D(None, 512, 3, pad=1),
            conv5_3=L.DilatedConvolution2D(None, 512, 3, pad=1),

            conv6=L.DilatedConvolution2D(None, 1024, 3, pad=6, dilate=6),
            conv7=L.Convolution2D(None, 1024, 1),

            conv8_1=L.Convolution2D(None, 256, 1),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1),

            conv9_1=L.Convolution2D(None, 128, 1),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1),

            conv10_1=L.Convolution2D(None, 128, 1),
            conv10_2=L.Convolution2D(None, 256, 3),

            conv11_1=L.Convolution2D(None, 128, 1),
            conv11_2=L.Convolution2D(None, 256, 3),

            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )

        for ar in self.aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(None, n * 4, 3, pad=1))
            self.conf.add_link(L.Convolution2D(
                None, n * (self.n_classes + 1), 3, pad=1))

        if pretrained_model == 'auto':
            _retrieve(
                'VGG_VOC0712_SSD_300.npz',
                'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel',
                self)
        elif pretrained_model:
            serializers.load_npz(pretrained_model, self)

        # the format of default_bbox is (center_x, center_y, width, height)
        self.default_bbox = self._default_bbox()

    def _default_bbox(self):
        bbox = list()
        for k in range(len(self.grids)):
            for v, u in itertools.product(range(self.grids[k]), repeat=2):
                cx = (u + 0.5) * self.steps[k]
                cy = (v + 0.5) * self.steps[k]

                s = self.sizes[k]
                bbox.append((cx, cy, s, s))

                s = np.sqrt(self.sizes[k] * self.sizes[k + 1])
                bbox.append((cx, cy, s, s))

                s = self.sizes[k]
                for ar in self.aspect_ratios[k]:
                    bbox.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    bbox.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        return np.stack(bbox)

    def __call__(self, x):
        hs = list()

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        hs.append(self.norm4(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)

        hs_loc = list()
        hs_conf = list()
        for i, x in enumerate(hs):
            h_loc = self.loc[i](x)
            h_loc = F.transpose(h_loc, (0, 2, 3, 1))
            h_loc = F.reshape(h_loc, (h_loc.shape[0], -1, 4))
            hs_loc.append(h_loc)

            h_conf = self.conf[i](x)
            h_conf = F.transpose(h_conf, (0, 2, 3, 1))
            h_conf = F.reshape(
                h_conf, (h_conf.shape[0], -1, self.n_classes + 1))
            hs_conf.append(h_conf)

        h_loc = F.concat(hs_loc, axis=1)
        h_conf = F.concat(hs_conf, axis=1)
        return h_loc, h_conf

    def _decode(self, loc, conf):
        # the format of bbox is (center_x, center_y, width, height)
        bbox = np.hstack((
            self.default_bbox[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_bbox[:, 2:],
            self.default_bbox[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        # convert the format of bbox to (x_min, y_min, x_max, y_max)
        bbox[:, :2] -= bbox[:, 2:] / 2
        bbox[:, 2:] += bbox[:, :2]
        conf = np.exp(conf)
        conf /= conf.sum(axis=1, keepdims=True)
        return bbox, conf

    def predict(self, img):
        H, W = img.shape[1:]
        img = transforms.resize(img, (self.insize, self.insize))
        img -= np.array(self.mean)[:, np.newaxis, np.newaxis]
        loc, conf = self(img[np.newaxis])
        bbox, conf = self._decode(loc.data[0], conf.data[0])
        bbox = transforms.resize_bbox(bbox, (1, 1), (W, H))
        return bbox, conf

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        caffemodel = _CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None)

        chainermodel.conv1_1.copyparams(caffemodel.conv1_1)
        chainermodel.conv1_2.copyparams(caffemodel.conv1_2)

        chainermodel.conv2_1.copyparams(caffemodel.conv2_1)
        chainermodel.conv2_2.copyparams(caffemodel.conv2_2)

        chainermodel.conv3_1.copyparams(caffemodel.conv3_1)
        chainermodel.conv3_2.copyparams(caffemodel.conv3_2)
        chainermodel.conv3_3.copyparams(caffemodel.conv3_3)

        chainermodel.conv4_1.copyparams(caffemodel.conv4_1)
        chainermodel.conv4_2.copyparams(caffemodel.conv4_2)
        chainermodel.conv4_3.copyparams(caffemodel.conv4_3)
        chainermodel.norm4.copyparams(caffemodel.conv4_3_norm)

        chainermodel.conv5_1.copyparams(caffemodel.conv5_1)
        chainermodel.conv5_2.copyparams(caffemodel.conv5_2)
        chainermodel.conv5_3.copyparams(caffemodel.conv5_3)

        chainermodel.conv6.copyparams(caffemodel.fc6)
        chainermodel.conv7.copyparams(caffemodel.fc7)

        chainermodel.conv8_1.copyparams(caffemodel.conv6_1)
        chainermodel.conv8_2.copyparams(caffemodel.conv6_2)

        chainermodel.conv9_1.copyparams(caffemodel.conv7_1)
        chainermodel.conv9_2.copyparams(caffemodel.conv7_2)

        chainermodel.conv10_1.copyparams(caffemodel.conv8_1)
        chainermodel.conv10_2.copyparams(caffemodel.conv8_2)

        chainermodel.conv11_1.copyparams(caffemodel.conv9_1)
        chainermodel.conv11_2.copyparams(caffemodel.conv9_2)

        chainermodel.loc[0].copyparams(caffemodel.conv4_3_norm_mbox_loc)
        chainermodel.conf[0].copyparams(caffemodel.conv4_3_norm_mbox_conf)

        chainermodel.loc[1].copyparams(caffemodel.fc7_mbox_loc)
        chainermodel.conf[1].copyparams(caffemodel.fc7_mbox_conf)

        chainermodel.loc[2].copyparams(caffemodel.conv6_2_mbox_loc)
        chainermodel.conf[2].copyparams(caffemodel.conv6_2_mbox_conf)

        chainermodel.loc[3].copyparams(caffemodel.conv7_2_mbox_loc)
        chainermodel.conf[3].copyparams(caffemodel.conv7_2_mbox_conf)

        chainermodel.loc[4].copyparams(caffemodel.conv8_2_mbox_loc)
        chainermodel.conf[4].copyparams(caffemodel.conv8_2_mbox_conf)

        chainermodel.loc[5].copyparams(caffemodel.conv9_2_mbox_loc)
        chainermodel.conf[5].copyparams(caffemodel.conv9_2_mbox_conf)

        serializers.save_npz(path_npz, chainermodel, compression=False)


def _make_npz(path_npz, path_caffemodel, model):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/weiliu89/caffe/tree/ssd\', '
            'and place it on {}'.format(path_caffemodel))
    SSD300.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    serializers.load_npz(path_npz, model)
    return model


def _retrieve(name_npz, name_caffemodel, model):
    root = download.get_dataset_directory('pfnet/chainercv/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return download.cache_or_load_file(
        path,
        lambda path: _make_npz(path, path_caffemodel, model),
        lambda path: serializers.load_npz(path, model))
