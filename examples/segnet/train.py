try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import numpy as np

from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from chainercv.datasets import CamVidDataset
from chainercv.datasets import TransformDataset
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.links import SegNetBasic


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--class_weight', type=str, default='class_weight.npy')
    parser.add_argument('--out', type=str, default='result')
    args = parser.parse_args()

    # Triggers
    log_trigger = (50, 'iteration')
    validation_trigger = (2000, 'iteration')
    end_trigger = (16000, 'iteration')

    # Dataset
    train = CamVidDataset(split='train')

    def transform(in_data):
        img, label = in_data
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1]
            label = label[:, ::-1]
        return img, label

    train = TransformDataset(train, transform)
    val = CamVidDataset(split='val')

    # Iterator
    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = iterators.MultiprocessIterator(
        val, args.batchsize, shuffle=False, repeat=False)

    # Model
    class_weight = np.load(args.class_weight)
    model = SegNetBasic(n_class=11)
    model = PixelwiseSoftmaxClassifier(
        model, class_weight=class_weight)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    # Trainer
    trainer = training.Trainer(updater, end_trigger, out=args.out)

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(TestModeEvaluator(val_iter, model,
                                     device=args.gpu),
                   trigger=validation_trigger)

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], x_key='iteration',
            file_name='loss.png'))

    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_iteration-{.updater.iteration}'),
        trigger=end_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr',
         'main/loss', 'validation/main/loss']),
        trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
