import argparse
import multiprocessing

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

import chainermn

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512

from train import Transform


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3, comm=None):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k
        self.comm = comm

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k, self.comm)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--test_batchsize', type=int, default=32)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    if comm.rank == 0:
        train = TransformDataset(
            ConcatenatedDataset(
                VOCBboxDataset(year='2007', split='trainval'),
                VOCBboxDataset(year='2012', split='trainval')
            ),
            Transform(model.coder, model.insize, model.mean))
        test = VOCBboxDataset(
            year='2007', split='test',
            use_difficult=True, return_difficult=True)
        test_iter = chainer.iterators.SerialIterator(
            test, args.test_batchsize, repeat=False, shuffle=False)
    else:
        train = None

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    multiprocessing.set_start_method('forkserver')
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize // comm.size, n_processes=2)

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(), comm)
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    if comm.rank == 0:
        trainer.extend(
            DetectionVOCEvaluator(
                test_iter, model, use_07_metric=True,
                label_names=voc_bbox_label_names),
            trigger=(10000, 'iteration'))

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
            extensions.snapshot_object(
                model, 'model_iter_{.updater.iteration}'),
            trigger=(120000, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
