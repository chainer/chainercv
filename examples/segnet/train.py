import argparse
from collections import defaultdict
import os

import chainer
import numpy as np

from chainer.dataset import concat_examples
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.links import SegNetBasic

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def recalculate_bn_statistics(model, batchsize):
    train = CamVidDataset(split='train')
    it = chainer.iterators.SerialIterator(
        train, batchsize, repeat=False, shuffle=False)
    bn_avg_mean = defaultdict(np.float32)
    bn_avg_var = defaultdict(np.float32)

    n_iter = 0
    for batch in it:
        imgs, _ = concat_examples(batch)
        model(model.xp.array(imgs))
        for name, link in model.namedlinks():
            if name.endswith('_bn'):
                bn_avg_mean[name] += link.avg_mean
                bn_avg_var[name] += link.avg_var
        n_iter += 1

    for name, link in model.namedlinks():
        if name.endswith('_bn'):
            link.avg_mean = bn_avg_mean[name] / n_iter
            link.avg_var = bn_avg_var[name] / n_iter

    return model


def transform(in_data):
    img, label = in_data
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    return img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--class-weight', type=str, default='class_weight.npy')
    parser.add_argument('--out', type=str, default='result')
    args = parser.parse_args()

    # Triggers
    log_trigger = (50, 'iteration')
    validation_trigger = (2000, 'iteration')
    end_trigger = (16000, 'iteration')

    # Dataset
    train = CamVidDataset(split='train')
    train = TransformDataset(train, transform)
    val = CamVidDataset(split='val')

    # Iterator
    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = iterators.MultiprocessIterator(
        val, args.batchsize, shuffle=False, repeat=False)

    # Model
    class_weight = np.load(args.class_weight)
    model = SegNetBasic(
        pretrained_model=None, **SegNetBasic.preset_params['camvid'])
    model = PixelwiseSoftmaxClassifier(
        model, class_weight=class_weight)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))

    # Updater
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    # Trainer
    trainer = training.Trainer(updater, end_trigger, out=args.out)

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss'], x_key='iteration',
            file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['validation/main/miou'], x_key='iteration',
            file_name='miou.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr',
         'main/loss', 'validation/main/miou',
         'validation/main/mean_class_accuracy',
         'validation/main/pixel_accuracy']),
        trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        SemanticSegmentationEvaluator(
            val_iter, model.predictor,
            camvid_label_names),
        trigger=validation_trigger)

    trainer.run()

    chainer.serializers.save_npz(
        os.path.join(args.out, 'snapshot_model.npz'),
        recalculate_bn_statistics(model.predictor, 24))


if __name__ == '__main__':
    main()
