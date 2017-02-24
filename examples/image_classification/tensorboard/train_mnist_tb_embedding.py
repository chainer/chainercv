#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import os.path as osp

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer_cv.extensions import EmbedImages
from chainer_cv.visualizations import embedding_tensorboard


# Network definition
class EmbedModel(chainer.Chain):

    def __init__(self, n_units):
        super(EmbedModel, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return h2


class MLP(chainer.Chain):

    def __init__(self, embed_model, n_out):
        super(MLP, self).__init__(
            embed=embed_model,
            l=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h = self.embed(x)
        return self.l(h)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    embed_model = EmbedModel(args.unit)
    mlp_model = MLP(embed_model, 10)
    model = L.Classifier(mlp_model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # embed features
    embed_file = 'embed.npy'
    trainer.extend(
        EmbedImages(test_iter, embed_model, filename=embed_file),
        trigger=(args.epoch, 'epoch'))

    # Run the training
    trainer.run()

    features = np.load(osp.join(trainer.out, embed_file))
    images = [data[0].reshape(28, 28) for data in test]
    images = np.stack(images)
    labels = {'label': [data[1] for data in test], 'id': range(len(test))}
    embedding_tensorboard(features, images, labels)


if __name__ == '__main__':
    main()
