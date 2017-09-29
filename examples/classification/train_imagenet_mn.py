from __future__ import division
import matplotlib
matplotlib.use('agg')
import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer.links import Classifier
from chainer import training
from chainer.training import extensions

from chainercv.datasets import DirectoryParsingLabelDataset

from chainercv.transforms import center_crop
from chainercv.transforms import pca_lighting
from chainercv.transforms import random_flip
from chainercv.transforms import random_sized_crop
from chainercv.transforms import resize
from chainercv.transforms import scale

from chainercv.datasets import directory_parsing_label_names

from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet18
from chainercv.links import ResNet50

import chainermn


class TrainTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua#L80
        img, label = in_data
        _, H, W = img.shape
        img = random_sized_crop(img)
        img = resize(img, (224, 224))
        img = random_flip(img, x_random=True)
        img = pca_lighting(img, 25)
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
    archs = {
        'resnet18': {'class': ResNet18, 'score_layer_name': 'fc6',
                     'kwargs': {'fb_resnet': True}},
        'resnet50': {'class': ResNet50, 'score_layer_name': 'fc6',
                     'kwargs': {'fb_resnet': True}},
        'resnet101': {'class': ResNet101, 'score_layer_name': 'fc6',
                      'kwargs': {'fb_resnet': True}},
        'resnet152': {'class': ResNet152, 'score_layer_name': 'fc6',
                      'kwargs': {'fb_resnet': True}}
    }
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to root of the train dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--arch',
                        '-a', choices=archs.keys(), default='resnet18',
                        help='Convnet architecture')
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--pretrained_model')
    # parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--loaderjob', type=int, default=4)
    parser.add_argument('--batchsize', type=int, default=64,
                        help='Batch size for each worker')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=90)
    args = parser.parse_args()

    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    label_names = directory_parsing_label_names(args.train)

    arch = archs[args.arch]
    extractor = arch['class'](n_class=len(label_names), **arch['kwargs'])
    extractor.pick = arch['score_layer_name']
    model = Classifier(extractor)

    if comm.rank == 0:
        train_data = DirectoryParsingLabelDataset(args.train)
        val_data = DirectoryParsingLabelDataset(args.val)
        train_data = TransformDataset(train_data, TrainTransform(extractor.mean))
        val_data = TransformDataset(val_data, ValTransform(extractor.mean))
        print('finished loading dataset')
    else:
        train_data, val_data = None, None
    train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)
    val_data = chainermn.scatter_dataset(val_data, comm, shuffle=True)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, args.batchsize, shared_mem=3 * 224 * 224 * 4)
    val_iter = iterators.MultiprocessIterator(
        val_data, args.batchsize,
        repeat=False, shuffle=False, shared_mem=3 * 224 * 224 * 4)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(
        lr=args.lr, momentum=args.momentum), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=device)

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'epoch'))
    evaluator = chainermn.create_multi_node_evaluator(
        extensions.Evaluator(val_iter, model, device=device), comm)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    log_interval = 0.1, 'epoch'
    print_interval = 0.1, 'epoch'
    plot_interval = 1, 'epoch'

    if comm.rank == 0:
        trainer.extend(
            extensions.snapshot_object(extractor, 'snapshot_model.npz'),
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


        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
