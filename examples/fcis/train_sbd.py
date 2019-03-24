from __future__ import division

import argparse
import numpy as np
import six

import chainer
from chainer.dataset.convert import _concat_arrays
from chainer.dataset.convert import to_device
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger

from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.datasets import SBDInstanceSegmentationDataset
from chainercv.experimental.links import FCISResNet101
from chainercv.experimental.links import FCISTrainChain
from chainercv.extensions import InstanceSegmentationVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv import transforms
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def concat_examples(batch, device=None):
    # batch: img, mask, label, scale
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]
    result = []
    for i in six.moves.range(len(first_elem)):
        array = _concat_arrays([example[i] for example in batch], None)
        if i == 0:  # img
            result.append(to_device(device, array))
        else:
            result.append(array)
    return tuple(result)


class Transform(object):

    def __init__(self, fcis):
        self.fcis = fcis

    def __call__(self, in_data):
        img, mask, label = in_data
        bbox = mask_to_bbox(mask)
        _, orig_H, orig_W = img.shape
        img = self.fcis.prepare(img)
        _, H, W = img.shape
        scale = H / orig_H
        mask = transforms.resize(mask.astype(np.float32), (H, W))
        bbox = transforms.resize_bbox(bbox, (orig_H, orig_W), (H, W))

        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        mask = transforms.flip(mask, x_flip=params['x_flip'])
        bbox = transforms.flip_bbox(bbox, (H, W), x_flip=params['x_flip'])
        return img, mask, label, bbox, scale


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: FCIS')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=0.0005)
    parser.add_argument(
        '--lr-cooldown-factor', '-lcf', type=float, default=0.1)
    parser.add_argument('--epoch', '-e', type=int, default=42)
    parser.add_argument('--cooldown-epoch', '-ce', type=int, default=28)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # dataset
    train_dataset = SBDInstanceSegmentationDataset(split='train')
    test_dataset = SBDInstanceSegmentationDataset(split='val')

    # model
    param = FCISResNet101.preset_param('sbd')
    param['iter2'] = False
    fcis = FCISResNet101(
        pretrained_model='imagenet', **param)
    fcis.use_preset('evaluate')
    model = FCISTrainChain(fcis)

    # gpu
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)

    model.fcis.head.conv1.W.update_rule.add_hook(GradientScaling(3.0))
    model.fcis.head.conv1.b.update_rule.add_hook(GradientScaling(3.0))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    for param in model.params():
        if param.name in ['beta', 'gamma']:
            param.update_rule.enabled = False
    model.fcis.extractor.conv1.disable_update()
    model.fcis.extractor.res2.disable_update()

    train_dataset = TransformDataset(
        train_dataset, Transform(model.fcis))

    # iterator
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=1)
    test_iter = chainer.iterators.SerialIterator(
        test_dataset, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, converter=concat_examples,
        device=args.gpu)

    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    # lr scheduler
    trainer.extend(
        chainer.training.extensions.ExponentialShift(
            'lr', args.lr_cooldown_factor, init=args.lr),
        trigger=(args.cooldown_epoch, 'epoch'))

    # interval
    log_interval = 100, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 20, 'iteration'

    # training extensions
    trainer.extend(
        extensions.snapshot_object(
            model.fcis, filename='snapshot_model.npz'),
        trigger=(args.epoch, 'epoch'))
    trainer.extend(
        extensions.observe_lr(),
        trigger=log_interval)
    trainer.extend(
        extensions.LogReport(log_name='log.json', trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'iteration', 'epoch', 'elapsed_time', 'lr',
        'main/loss',
        'main/rpn_loc_loss',
        'main/rpn_cls_loss',
        'main/roi_loc_loss',
        'main/roi_cls_loss',
        'main/roi_mask_loss',
        'validation/main/map',
    ]), trigger=print_interval)
    trainer.extend(
        extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval),
            trigger=plot_interval)

    trainer.extend(
        InstanceSegmentationVOCEvaluator(
            test_iter, model.fcis,
            iou_thresh=0.5, use_07_metric=True,
            label_names=sbd_instance_segmentation_label_names),
        trigger=ManualScheduleTrigger(
            [len(train_dataset) * args.cooldown_epoch,
             len(train_dataset) * args.epoch], 'iteration'))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
