import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

from chainer_cv.datasets import VOCDetectionDataset
from chainer_cv.wrappers import ResizeWrapper
from chainer_cv.wrappers import RandomMirrorWrapper
from chainer_cv.wrappers import output_shape_soft_min_hard_max
from chainer_cv.wrappers import bbox_resize_hook
from chainer_cv.wrappers import bbox_mirror_hook
from chainer_cv.wrappers import SubtractWrapper

from faster_rcnn import FasterRCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-ba', '--batch-size', type=int, default=1)
    parser.add_argument('-l', '--lr', type=float, default=5e-4)
    parser.add_argument('-o', '--out', type=str, default='result')

    args = parser.parse_args()
    epoch = args.epoch
    gpu = args.gpu
    batch_size = args.batch_size
    lr = args.lr
    out = args.out

    train_data = VOCDetectionDataset(mode='train', use_cache=True, year='2007')
    test_data = VOCDetectionDataset(mode='val', use_cache=True, year='2007')

    wrappers = [
        lambda d: SubtractWrapper(
            d, value=np.array([103.939, 116.779, 123.68])),
        lambda d: ResizeWrapper(
            d, preprocess_idx=0,
            output_shape=output_shape_soft_min_hard_max(600, 1200),
            hook=bbox_resize_hook(1)),
        lambda d: RandomMirrorWrapper(d, augment_idx=0, orientation='h',
                                      hook=bbox_mirror_hook())
    ]
    for wrapper in wrappers:
        train_data = wrapper(train_data)
        test_data = wrapper(test_data)

    model = FasterRCNN()
    # optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    optimizer = chainer.optimizers.Adam(
        alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    train_iter = chainer.iterators.SerialIterator(test_data, batch_size=1)

    updater = chainer.training.updater.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    log_interval = 1, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'main/time',
         'main/rpn_loss_cls',
         'main/rpn_loss_bbox',
         'main/loss_bbox',
         'main/loss_cls']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()
