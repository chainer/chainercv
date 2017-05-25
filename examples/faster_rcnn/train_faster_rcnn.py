from __future__ import division

import matplotlib
matplotlib.use('agg')
import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

from chainercv.datasets import TransformDataset
from chainercv.datasets import VOCDetectionDataset
from chainercv import transforms

from chainercv.datasets import voc_detection_label_names
from chainercv.links import FasterRCNNTrainChain
from chainercv.links import FasterRCNNVGG16


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: Faster R-CNN')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    labels = voc_detection_label_names
    train_data = VOCDetectionDataset(split='trainval', year='2007')
    faster_rcnn = FasterRCNNVGG16(n_fg_class=len(labels),
                                  pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    def transform(in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (W, H), (o_W, o_H))

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(bbox, (o_W, o_H), params['x_flip'])

        return img, bbox, label, scale
    train_data = TransformDataset(train_data, transform)

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn, 'snapshot_model.npz'),
        trigger=(args.iteration, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'iteration'))

    log_interval = 20, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/loc_loss',
         'main/cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'map'
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
