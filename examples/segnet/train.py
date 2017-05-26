import matplotlib  # isort:skip # NOQA
matplotlib.use('Agg')  # isort:skiip # NOQA

import argparse
import time

import chainer
import numpy as np

from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainercv.datasets import CamVidDataset
from chainercv.datasets import TransformDataset
from chainercv.links.loss import PixelwiseSoftmaxWithWeightClassifier
from chainercv.links.model.segnet import segnet_basic


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--batchsize', type=int, default=12)
parser.add_argument('--class_weight', type=str, default='class_weight.npy')
args = parser.parse_args()

# Dataset
train = CamVidDataset(mode='train')


def transform(in_data):
    x, t = in_data
    if np.random.rand() > 0.5:
        x = x[:, :, ::-1]
        t = t[:, :, ::-1]
    return x, t


train = TransformDataset(train, transform)
val = CamVidDataset(mode='val')

# Iterator
train_iter = iterators.MultiprocessIterator(train, args.batchsize)
val_iter = iterators.MultiprocessIterator(
    val, args.batchsize, shuffle=False, repeat=False)

# Model
class_weight = np.load(args.class_weight)[:11]
model = segnet_basic.SegNetBasic(out_ch=11)
model = PixelwiseSoftmaxWithWeightClassifier(model, 11, 11, class_weight)

# Optimizer
optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

# Updater
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)


# Trainer
trainer = training.Trainer(
    updater, (140000, 'iteration'),
    out='result/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S')))


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


report_trigger = (1000, 'iteration')
trainer.extend(extensions.LogReport(trigger=report_trigger))
trainer.extend(extensions.observe_lr(), trigger=report_trigger)
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(TestModeEvaluator(val_iter, model,
                                 device=args.gpu), trigger=report_trigger)
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'main/loss', 'main/mean_iou',
     'main/mean_pixel_accuracy', 'validation/main/loss',
     'validation/main/mean_iou', 'validation/main/mean_pixel_accuracy',
     'elapsed_time', 'lr']),
    trigger=report_trigger)
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'], x_key='iteration',
    file_name='loss.png'))
trainer.extend(extensions.PlotReport(
    ['main/mean_iou', 'validation/main/mean_iou'], x_key='iteration',
    file_name='mean_iou.png'))
trainer.extend(extensions.PlotReport(
    ['main/mean_pixel_accuracy', 'validation/main/mean_pixel_accuracy'],
    x_key='iteration', file_name='mean_pixel_accuracy.png'))
trainer.extend(extensions.snapshot(
    filename='snapshot_iteration-{.updater.iteration}'),
    trigger=report_trigger)
trainer.extend(extensions.snapshot_object(
    model.predictor, filename='model_iteration-{.updater.iteration}',
    trigger=report_trigger))
trainer.extend(extensions.ProgressBar(), trigger=report_trigger)

trainer.run()
