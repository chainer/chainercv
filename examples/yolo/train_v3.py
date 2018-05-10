from __future__ import division

import argparse
import numpy as np

import chainer
from chainer.backends import cuda
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
import chainer.functions as F
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.yolo import ManualShift
from chainercv.links import YOLOv3
from chainercv import transforms
from chainercv import utils


class TrainChain(chainer.Chain):

    def __init__(self, model):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, imgs, masks, gt_locs, gt_objs, gt_labels):
        locs, objs, confs = self.model(imgs)
        loc_loss = F.mean(
            F.squared_error(locs, gt_locs)
            * F.broadcast_to(objs[:, :, None], gt_locs.shape))
        obj_loss = F.mean(
            F.sigmoid_cross_entropy(objs, gt_objs, reduce='no') * masks)
        conf_loss = F.mean(
            F.sigmoid_cross_entropy(confs, gt_labels, reduce='no')
            * F.broadcast_to(objs[:, :, None], gt_labels.shape))

        loss = loc_loss + obj_loss + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss,
             'loss/obj': obj_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self,  model):
        self._n_fg_class = model.n_fg_class
        self._insize = model.insize
        self._default_bbox = cuda.to_cpu(model._default_bbox)
        self._step = cuda.to_cpu(model._step)

    def _encode(self, bbox, label):
        if len(bbox) == 0:
            n_default_bbox = self._default_bbox.shape[0]
            mask = np.zeros(n_default_bbox, dtype=np.int32)
            loc = np.zeros((n_default_bbox, 4), dtype=np.float32)
            obj = np.zeros(n_default_bbox, dtype=np.int32)
            label = np.zeros(
                (n_default_bbox, self._n_fg_class), dtype=np.int32)
            return mask, loc, obj, label

        iou = utils.bbox_iou(
            np.hstack((
                (self._default_bbox[:, :2] + 1 / 2) * self._step[:, None]
                - self._default_bbox[:, 2:] / 2,
                (self._default_bbox[:, :2] + 1 / 2) * self._step[:, None]
                + self._default_bbox[:, 2:] / 2)),
            bbox)

        index = np.empty(len(self._default_bbox), dtype=int)
        index[:] = -1
        index[iou.max(axis=1) >= 0.5] = -2

        while True:
            i, j = np.unravel_index(iou.argmax(), iou.shape)
            if iou[i, j] < 0.5:
                break
            index[i] = j
            iou[i, :] = 0
            iou[:, j] = 0

        mask = (index >= -1).astype(np.int32)
        index[index < 0] = -1

        loc = bbox[index].copy()
        loc[:, 2:] -= loc[:, :2]
        loc[:, :2] += loc[:, 2:] / 2
        loc[:, :2] /= self._step[:, None]
        loc[:, :2] -= self._default_bbox[:, :2]
        loc[:, :2] = -np.log(1 / loc[:, :2] - 1)
        loc[:, 2:] /= self._default_bbox[:, 2:]
        loc[:, 2:] = np.log(loc[:, 2:])

        obj = (index >= 0).astype(np.int32)

        label = (label[index][:, None] == np.arange(self._n_fg_class)) \
            .astype(np.int32)

        return mask, loc, obj, label

    def __call__(self, in_data):
        img, bbox, label = in_data

        _, H, W = img.shape
        img, param = transforms.resize_contain(
            img / 255, (self._insize, self._insize), fill=0.5,
            return_param=True)
        bbox = transforms.resize_bbox(bbox, (H, W), param['scaled_size'])

        mask, loc, obj, label = self._encode(bbox, label)
        return img, mask, loc, obj, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    model = YOLOv3(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model='imagenet')

    model.use_preset('evaluate')
    train_chain = TrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train = TransformDataset(
        ConcatenatedDataset(
            VOCBboxDataset(year='2007', split='trainval'),
            VOCBboxDataset(year='2012', split='trainval')
        ),
        Transform(model))
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)

    test = VOCBboxDataset(
        year='2007', split='test',
        use_difficult=True, return_difficult=True)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # initial lr is set to 1e-4 by ManualShift
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    optimizer.add_hook(WeightDecay(0.0005))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (50200, 'iteration'), args.out)
    trainer.extend(
        ManualShift('lr', [1e-3, 1e-4, 1e-5], init=1e-4),
        trigger=triggers.ManualScheduleTrigger(
            [1000, 40000, 45000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=triggers.ManualScheduleTrigger(
            [10000, 20000, 30000, 40000, 50200], 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(50200, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
