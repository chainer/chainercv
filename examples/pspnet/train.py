import argparse
import copy
import cv2
cv2.setNumThreads(0)
import numpy as np

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.chainer_experimental.training import PolynomialShift
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.experimental.links import PSPNetResNet101
from chainercv.links import Conv2DBNActiv
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv import transforms

import PIL


class Transform(object):

    def __init__(
            self, mean,
            scale_range=[0.5, 2.0], crop_size=(713, 713),
            color_sigma=25.5):
        self.mean = mean
        self.scale_range = scale_range
        self.crop_size = crop_size
        self.color_sigma = color_sigma

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Scale
        scaled_H = int(scale * H)
        scaled_W = int(scale * W)
        img = transforms.resize(img, (scaled_H, scaled_W), PIL.Image.BICUBIC)
        label = transforms.resize(
            label[None], (scaled_H, scaled_W), PIL.Image.NEAREST)[0]

        # Crop
        if (scaled_H < self.crop_size[0]) or (scaled_W < self.crop_size[1]):
            shorter_side = min(img.shape[1:])
            img, param = transforms.random_crop(img, (shorter_side, shorter_side), True)
        else:
            img, param = transforms.random_crop(img, self.crop_size, True)
        label = label[param['y_slice'], param['x_slice']]

        # Rotate
        angle = np.random.uniform(-10, 10)
        H, W = img.shape[1:]
        img = img.transpose(1, 2, 0)
        r = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1)
        img = cv2.warpAffine(img, r, (W, H)).transpose(2, 0, 1)
        label = cv2.warpAffine(label, r, (W, H), flags=cv2.INTER_NEAREST,
                               borderValue=-1)

        # Resize
        if ((img.shape[1] < self.crop_size[0])
                or (img.shape[2] < self.crop_size[1])):
            img = transforms.resize(img, self.crop_size, PIL.Image.BICUBIC)
        if ((label.shape[0] < self.crop_size[0])
                or (label.shape[1] < self.crop_size[1])):
            label = transforms.resize(
                label[None].astype(np.float32), self.crop_size, PIL.Image.NEAREST)
            label = label.astype(np.int32)[0]

        # Horizontal flip
        if np.random.rand() > 0.5:
            img = transforms.flip(img, x_flip=True)
            label = transforms.flip(label[None], x_flip=True)[0]

        # Color augmentation
        img = transforms.pca_lighting(img, self.color_sigma)

        # Mean subtraction
        img = img - self.mean
        return img, label


class TrainChain(chainer.Chain):

    def __init__(self, model, ignore_label=-1):
        initialW = chainer.initializers.HeNormal()
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model
            self.aux_conv1 = Conv2DBNActiv(
                None, 512, 3, 1, 1, initialW=initialW)
            self.aux_conv2 = L.Convolution2D(
                None, model.n_class, 3, 1, 1, False, initialW=initialW)
        self.ignore_label = ignore_label

    def __call__(self, imgs, labels):
        h_aux, h_main = self.model.extractor(imgs)
        h_aux = F.dropout(self.aux_conv1(h_aux), ratio=0.1)
        h_aux = self.aux_conv2(h_aux)
        h_aux = F.resize_images(h_aux, imgs.shape[2:])

        h_main = model.ppm(h_main)
        h_main = F.dropout(self.model.head_conv1(h_main), ratio=0.1)
        h_main = self.model.head_conv2(h_main)
        h_main = F.resize_images(h_main, imgs.shape[2:])

        aux_loss = F.softmax_cross_entropy(
            h_aux, labels, ignore_label=self.ignore_label)
        main_loss = F.softmax_cross_entropy(
            h_main, labels, ignore_label=self.ignore_label)
        loss = 0.4 * aux_loss + main_loss

        chainer.reporter.report({'loss': loss}, self)
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    n_class = len(cityscapes_semantic_segmentation_label_names)

    model = PSPNetResNet101(n_class, input_size=(713, 713))
    train_chain = TrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        train_chain.to_gpu()

    n_iter = 90000

    train = TransformDataset(
        CityscapesSemanticSegmentationDataset(
            label_resolution='fine', split='train'),
        Transform(model.mean))
    val = CityscapesSemanticSegmentationDataset(
            label_resolution='fine', split='val')
    train_iter = chainer.iterators.MultiprocessIterator(
       train, batch_size=2)
    val_iter = chainer.iterators.SerialIterator(
        val, batch_size=1, repeat=False, shuffle=False)

    optimizer = chainer.optimizers.MomentumSGD(0.01, 0.9)
    optimizer.setup(train_chain)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.9))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (n_iter, 'iteration'), args.out)
    trainer.extend(
        PolynomialShift('lr', (0.9, n_iter), optimizer=optimizer),
        trigger=(1, 'iteration'))

    log_interval = 10, 'iteration'
    val_interval = 10000, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr', 'main/loss',
         'validation/main/miou', 'validation/main/mean_class_accuracy',
         'validation/main/pixel_accuracy']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        SemanticSegmentationEvaluator(
            val_iter, model,
            cityscapes_semantic_segmentation_label_names),
        trigger=val_interval)

    trainer.run()
