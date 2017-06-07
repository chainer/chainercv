import argparse
import copy
import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import VOCDetectionDataset
from chainercv.links.model.ssd import ConcatenatedDataset
from chainercv.links.model.ssd import multibox_loss
from chainercv.links.model.ssd import random_transform
from chainercv.links.model.ssd import SelectiveWeightDecay
from chainercv.links import SSD300
from chainercv import transforms


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__(model=model)
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    args = parser.parse_args()

    model = SSD300(n_fg_class=20)
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

    dataset = TransformDataset(
        ConcatenatedDataset(
            VOCDetectionDataset(year='2007', split='trainval'),
            VOCDetectionDataset(year='2012', split='trainval')
        ),
        transform)

    iterator = chainer.iterators.MultiprocessIterator(
        dataset, args.batchsize, n_processes=2)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    optimizer.add_hook(SelectiveWeightDecay(0.0005, b={'lr': 2, 'decay': 0}))

    updater = training.StandardUpdater(iterator, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=0.001),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    snapshot_interval = 1000, 'iteration'
    log_interval = 10, 'iteration'

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        [
            'epoch', 'iteration',
            'main/loss', 'main/loss/loc', 'main/loss/conf', 'lr']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
