from __future__ import division
import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer.links import Classifier
from chainer.optimizer import WeightDecay
from chainer import training
from chainer.training import extensions

from chainercv.datasets import DirectoryParsingLabelDataset

from chainercv.transforms import center_crop
from chainercv.transforms import random_flip
from chainercv.transforms import random_sized_crop
from chainercv.transforms import resize
from chainercv.transforms import scale

from chainercv.datasets import directory_parsing_label_names

from chainercv.chainer_experimental.optimizers import CorrectedMomentumSGD
from chainercv.links.model.resnet import Bottleneck
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50

import chainermn

import cv2
cv2.setNumThreads(2)


class TrainTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape
        img = random_sized_crop(img)
        img = resize(img, (224, 224))
        img = random_flip(img, x_random=True)
        img -= self.mean
        return img, label


class ValTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        img = scale(img, 256)
        img = center_crop(img, (224, 224))
        img -= self.mean
        return img, label


import multiprocessing


def main():
    archs = {
        'resnet50': {'class': ResNet50, 'score_layer_name': 'fc6',
                     'kwargs': {'arch': 'fb'}},
        'resnet101': {'class': ResNet101, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}},
        'resnet152': {'class': ResNet152, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}}
    }
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to root of the train dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--arch',
                        '-a', choices=archs.keys(), default='resnet50',
                        help='Convnet architecture')
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--loaderjob', type=int, default=4)
    parser.add_argument('--batchsize', type=int, default=32,
                        help='Batch size for each worker')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--epoch', type=int, default=90)
    args = parser.parse_args()

    # We need to change the start method of multiprocessing module if we are
    # using InfiniBand and MultiprocessIterator. This is because processes
    # often crash when calling fork if they are using Infiniband.
    # (c.f., https://www.open-mpi.org/faq/?category=tuning#fork-warning )
    # Also, just setting the start method does not seem to be sufficient
    # to actually launch the forkserver, so also start a dummy process.
    # This must be done *before* calling `chainermn.create_communicator`!!!
    multiprocessing.set_start_method('forkserver')
    # TODO make this silent
    p = multiprocessing.Process(target=print, args=('Initialize forkserver',))
    p.start()
    p.join()

    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    if args.lr is not None:
        lr = args.lr
    else:
        lr = 0.1 * (args.batchsize * comm.size) / 256
        if comm.rank == 0:
            print('lr={}: lr is selected based on the linear '
                  'scaling rule'.format(lr))

    label_names = directory_parsing_label_names(args.train)

    arch = archs[args.arch]
    extractor = arch['class'](n_class=len(label_names), **arch['kwargs'])
    extractor.pick = arch['score_layer_name']
    model = Classifier(extractor)
    # Following https://arxiv.org/pdf/1706.02677.pdf,
    # the gamma of the last BN of each resblock is initialized by zeros.
    for l in model.links():
        if isinstance(l, Bottleneck):
            l.conv3.bn.gamma.data[:] = 0

    if comm.rank == 0:
        train_data = DirectoryParsingLabelDataset(args.train)
        val_data = DirectoryParsingLabelDataset(args.val)
        train_data = TransformDataset(
            train_data, TrainTransform(extractor.mean))
        val_data = TransformDataset(val_data, ValTransform(extractor.mean))
        print('finished loading dataset')
    else:
        train_data, val_data = None, None
    train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)
    val_data = chainermn.scatter_dataset(val_data, comm, shuffle=True)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, args.batchsize, shared_mem=3 * 224 * 224 * 4,
        n_processes=args.loaderjob)
    val_iter = iterators.MultiprocessIterator(
        val_data, args.batchsize,
        repeat=False, shuffle=False, shared_mem=3 * 224 * 224 * 4,
        n_processes=args.loaderjob)

    optimizer = chainermn.create_multi_node_optimizer(
        CorrectedMomentumSGD(lr=lr, momentum=args.momentum), comm)
    optimizer.setup(model)
    for param in model.params():
        if param.name not in ('beta', 'gamma'):
            param.update_rule.add_hook(WeightDecay(args.weight_decay))

    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    # Configure GPU setting
    chainer.cuda.set_max_workspace_size(1 * 1024 * 1024 * 1024)
    chainer.using_config('autotune', True)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=device)

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    warmup_iter = 5 * len(train_data) // args.batchsize  # 5 epochs
    warmup_mult = min((8 / comm.size, 1))
    trainer.extend(
        extensions.LinearShift(
            'lr', value_range=(lr * warmup_mult, lr),
            time_range=(0, warmup_iter)),
        trigger=chainer.training.triggers.ManualScheduleTrigger(
            list(range(warmup_iter + 1)), 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1, init=lr),
                   trigger=chainer.training.triggers.ManualScheduleTrigger(
                       [30, 60, 80], 'epoch'))
    evaluator = chainermn.create_multi_node_evaluator(
        extensions.Evaluator(val_iter, model, device=device), comm)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    log_interval = 0.1, 'epoch'
    print_interval = 0.1, 'epoch'
    plot_interval = 1, 'epoch'

    if comm.rank == 0:
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
        trainer.extend(
            extensions.snapshot_object(
                extractor, 'snapshot_model_{.updater.epoch}.npz'),
            trigger=(args.epoch, 'epoch'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'epoch', 'elapsed_time', 'lr',
             'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy']
        ), trigger=print_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(
                    ['main/loss', 'validation/main/loss'],
                    file_name='loss.png', trigger=plot_interval
                ),
                trigger=plot_interval
            )
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    file_name='accuracy.png', trigger=plot_interval
                ),
                trigger=plot_interval
            )

    trainer.run()


if __name__ == '__main__':
    main()
