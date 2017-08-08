#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from collections import namedtuple
from functools import partial

import numpy as np
from PIL import Image

import cv2 as cv
from chainer import dataset
from chainer import datasets
from chainercv import transforms

Label = namedtuple(
    'Label', ['name', 'id', 'trainId', 'category', 'categoryId',
              'hasInstances', 'ignoreInEval', 'color'])


# NOTE: The colors are in RGB order.
labels = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('egovehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectificationborder', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('outofroi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('railtrack', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guardrail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('trafficlight', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('trafficsign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('licenseplate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


class Cityscapes(dataset.DatasetMixin):

    """Cityscapes Dataset

    The mean file is in BGR order.
    All image files are also read in BGR order.

    """

    def __init__(self, img_dir, label_dir, split='train'):
        img_dir = os.path.join(img_dir, split)
        resol = os.path.basename(label_dir)

        self.label_fns = []
        for dname in glob.glob('{}/*'.format(label_dir)):
            if split in dname:
                for label_fn in glob.glob('{}/*/*_labelIds.png'.format(dname)):
                    self.label_fns.append(label_fn)

        self.img_fns = []
        for label_fn in self.label_fns:
            img_fn = label_fn.replace(resol, 'leftImg8bit')
            img_fn = img_fn.replace('_labelIds', '')
            self.img_fns.append(img_fn)

    def __len__(self):
        return len(self.img_fns)

    def get_example(self, i):
        assert os.path.exists(self.img_fns[i])
        assert os.path.exists(self.label_fns[i])
        img = cv.imread(self.img_fns[i])
        label_orig = cv.imread(self.label_fns[i], cv.IMREAD_GRAYSCALE)
        H, W = label_orig.shape
        label_out = np.ones((H, W)) * -1
        for label in labels:
            if label.ignoreInEval:
                label_out[np.where(label_orig == label.id)] = -1
            else:
                label_out[np.where(label_orig == label.id)] = label.trainId
        img = img.astype(np.float32).transpose(2, 0, 1)
        return img, label_out.astype(np.int32)


def _transform(inputs, mean=None, crop_size=(512, 512),
               scale=[0.5, 2.0], rotate=False, fliplr=False,
               ignore_labels=[19], label_size=None, n_class=20):
    img, label = inputs

    # Scaling
    if scale:
        if isinstance(scale, (list, tuple)):
            scale = np.random.uniform(scale[0], scale[1])
        scaled_h = int(img.shape[1] * scale)
        scaled_w = int(img.shape[2] * scale)
        img = transforms.resize(img, (scaled_h, scaled_w), Image.BICUBIC)
        label = transforms.resize(
            label[None, ...], (scaled_h, scaled_w), Image.NEAREST)[0]

    # Crop
    if crop_size is not None:
        if (img.shape[1] < crop_size[0]) or (img.shape[2] < crop_size[1]):
            shorter_side = min(img.shape[1:])
            _crop_size = (shorter_side, shorter_side)
            img, param = transforms.random_crop(img, _crop_size, True)
        else:
            img, param = transforms.random_crop(img, crop_size, True)
        label = label[param['y_slice'], param['x_slice']]

    # Rotate
    if rotate:
        angle = np.random.uniform(-10, 10)
        rows, cols = img.shape[1:]

        img = img.transpose(1, 2, 0)
        r = cv.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        img = cv.warpAffine(img, r, (cols, rows)).transpose(2, 0, 1)

        r = cv.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        label = cv.warpAffine(label, r, (cols, rows), flags=cv.INTER_NEAREST,
                              borderValue=-1)

    # Resize
    if crop_size is not None:
        if (img.shape[1] < crop_size[0]) or (img.shape[2] < crop_size[1]):
            img = transforms.resize(img, crop_size, Image.BICUBIC)

    if label_size is not None:
        label = transforms.resize(
            label[None, ...].astype(np.float32), label_size, Image.NEAREST)
        label = label.astype(np.int32)[0]
    else:
        if (label.shape[0] < crop_size[0]) or (label.shape[1] < crop_size[1]):
            label = transforms.resize(
                label[None, ...].astype(np.float32), crop_size, Image.NEAREST)
            label = label.astype(np.int32)[0]

    # Mean subtraction
    if mean is not None:
        img -= mean[:, None, None]

    # LR-flipping
    if fliplr:
        if np.random.rand() > 0.5:
            img = transforms.flip(img, x_flip=True)
            label = transforms.flip(label[None, ...], x_flip=True)[0]

    # Ignore label replacement
    for ignore_label in ignore_labels:
        label[np.where(label == ignore_label)] = -1

    assert label.max() < n_class, '{}'.format(label.max())
    if crop_size is not None:
        assert img.shape == (3, crop_size[0], crop_size[1]), \
            '{} != {}'.format(img.shape, crop_size)
        if label_size is None:
            assert label.shape == (crop_size[0], crop_size[1]), \
                '{} != {}'.format(label.shape, label_size)
        else:
            assert label.shape == (label_size[0], label_size[1]), \
                '{} != {}'.format(label.shape, label_size)

    return img, label


class TransformedCityscapes(datasets.TransformDataset):

    def __init__(self, img_dir, label_dir, split,
                 mean_fn=None, crop_size=(512, 512),
                 scale=[0.5, 2.0], rotate=False, fliplr=False,
                 ignore_labels=[19], label_size=None, n_class=20):
        self.d = Cityscapes(img_dir, label_dir, split)
        m = np.load(mean_fn) if mean_fn is not None else None
        if m is not None and m.ndim == 3:
            m = m.mean(axis=(0, 1))
        t = partial(
            _transform, mean=m, crop_size=crop_size,
            scale=scale, rotate=rotate, fliplr=fliplr,
            ignore_labels=ignore_labels, label_size=label_size,
            n_class=n_class)
        super().__init__(self.d, t)


def save_mean(img_dir, label_dir, out_dir):
    dataset = Cityscapes(img_dir, label_dir, 'train')
    resol = os.path.basename(label_dir)
    mean = None
    for img, _ in dataset:
        if mean is None:
            mean = img
        else:
            mean += img
    mean = mean / float(len(dataset))
    np.save(os.path.join(out_dir, 'train_mean_{}'.format(resol)), mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_dir', type=str, help='Path to "leftImg8bit" dir')
    parser.add_argument(
        '--label_dir', type=str, help='Path to "gtFine" or "gtCoarse" dir')
    parser.add_argument(
        '--out_dir', type=str, default='.',
        help='Path to the dir where the resulting mean.npy will be saved in')
    args = parser.parse_args()

    save_mean(args.img_dir, args.label_dir, args.out_dir)
