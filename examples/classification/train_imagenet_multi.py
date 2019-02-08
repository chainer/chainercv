from __future__ import division
import argparse
import multiprocessing
import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer.links import Classifier
from chainer.optimizer import WeightDecay
from chainer.optimizers import CorrectedMomentumSGD
from chainer import training
from chainer.training import extensions

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.transforms import center_crop
from chainercv.transforms import random_flip
from chainercv.transforms import random_sized_crop
from chainercv.transforms import resize
from chainercv.transforms import scale

from chainercv.chainer_experimental.training.extensions import make_shift

from chainercv.links.model.resnet import Bottleneck
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50

import chainermn

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


class TrainTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
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


def main():
    model_cfgs = {
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
    parser.add_argument('--model',
                        '-m', choices=model_cfgs.keys(), default='resnet50',
                        help='Convnet models')
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

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
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

    model_cfg = model_cfgs[args.model]
    extractor = model_cfg['class'](
        n_class=len(label_names), **model_cfg['kwargs'])
    extractor.pick = model_cfg['score_layer_name']
    model = Classifier(extractor)
    # Following https://arxiv.org/pdf/1706.02677.pdf,
    # the gamma of the last BN of each resblock is initialized by zeros.
    for l in model.links():
        if isinstance(l, Bottleneck):
            l.conv3.bn.gamma.data[:] = 0

    train_data = DirectoryParsingLabelDataset(args.train)
    val_data = DirectoryParsingLabelDataset(args.val)
    train_data = TransformDataset(
        train_data, TrainTransform(extractor.mean))
    val_data = TransformDataset(val_data, ValTransform(extractor.mean))
    print('finished loading dataset')

    if comm.rank == 0:
        train_indices = np.arange(len(train_data))
        val_indices = np.arange(len(val_data))
    else:
        train_indices = None
        val_indices = None

    train_indices = chainermn.scatter_dataset(
        train_indices, comm, shuffle=True)
    val_indices = chainermn.scatter_dataset(val_indices, comm, shuffle=True)
    train_data = train_data.slice[train_indices]
    val_data = val_data.slice[val_indices]
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, args.batchsize, n_processes=args.loaderjob)
    val_iter = iterators.MultiprocessIterator(
        val_data, args.batchsize,
        repeat=False, shuffle=False, n_processes=args.loaderjob)

    optimizer = chainermn.create_multi_node_optimizer(
        CorrectedMomentumSGD(lr=lr, momentum=args.momentum), comm)
    optimizer.setup(model)
    for param in model.params():
        if param.name not in ('beta', 'gamma'):
            param.update_rule.add_hook(WeightDecay(args.weight_decay))

    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=device)

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    @make_shift('lr')
    def warmup_and_exponential_shift(trainer):
        epoch = trainer.updater.epoch_detail
        warmup_epoch = 5
        if epoch < warmup_epoch:
            if lr > 0.1:
                warmup_rate = 0.1 / lr
                rate = warmup_rate \
                    + (1 - warmup_rate) * epoch / warmup_epoch
            else:
                rate = 1
        elif epoch < 30:
            rate = 1
        elif epoch < 60:
            rate = 0.1
        elif epoch < 80:
            rate = 0.01
        else:
            rate = 0.001
        return rate * lr

    trainer.extend(warmup_and_exponential_shift)
    evaluator = chainermn.create_multi_node_evaluator(
        extensions.Evaluator(val_iter, model, device=device), comm)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    log_interval = 0.1, 'epoch'
    print_interval = 0.1, 'epoch'

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

    trainer.run()


if __name__ == '__main__':
    main()
