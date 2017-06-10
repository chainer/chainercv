import argparse
import copy
import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import ConcatenatedDataset
from chainercv.links.model.ssd import MultiboxTrainChain
from chainercv.links.model.ssd import random_transform
from chainercv.links.model.ssd import SelectiveWeightDecay
from chainercv.links import SSD300
from chainercv import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    model = SSD300(
        n_fg_class=len(voc_detection_label_names),
        pretrained_model='imagenet')
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    coder = copy.copy(model.coder)
    coder.to_cpu()
    size = model.insize
    mean = model.mean

    def transform(in_data):
        img, bbox, label = in_data
        img, bbox, label = random_transform(img, bbox, label, size, mean)
        img -= np.array(mean)[:, np.newaxis, np.newaxis]
        mb_loc, mb_label = coder.encode(
            transforms.resize_bbox(bbox, (size, size), (1, 1)), label)
        return img, mb_loc, mb_label

    train = TransformDataset(
        ConcatenatedDataset(
            VOCDetectionDataset(year='2007', split='trainval'),
            VOCDetectionDataset(year='2012', split='trainval')
        ),
        transform)
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=2)

    test = VOCDetectionDataset(
        year='2007', split='test',
        use_difficult=True, return_difficult=True)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    optimizer.add_hook(SelectiveWeightDecay(0.0005, b={'lr': 2, 'decay': 0}))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_detection_label_names),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        [
            'epoch', 'iteration',
            'main/loss', 'main/loss/loc', 'main/loss/conf',
            'validation/main/map', 'lr']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(1000, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
