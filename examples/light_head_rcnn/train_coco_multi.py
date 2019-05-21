from __future__ import division

import argparse
import functools
import multiprocessing
import numpy as np
import random
import six

import chainer
from chainer.dataset.convert import _concat_arrays
from chainer.dataset.convert import to_device
import chainer.links as L
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger

from chainercv.chainer_experimental.datasets.sliceable \
    import TransformDataset
from chainercv.chainer_experimental.training.extensions import make_shift
from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.extensions import DetectionCOCOEvaluator
from chainercv.links.model.light_head_rcnn import LightHeadRCNNResNet101
from chainercv.links.model.light_head_rcnn import LightHeadRCNNTrainChain
from chainercv.links.model.ssd import GradientScaling
from chainercv import transforms
import chainermn


# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def concat_examples(batch, device=None, padding=None,
                    indices_concat=None, indices_to_device=None):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    elem_size = len(first_elem)
    if indices_concat is None:
        indices_concat = range(elem_size)
    if indices_to_device is None:
        indices_to_device = range(elem_size)

    result = []
    if not isinstance(padding, tuple):
        padding = [padding] * elem_size

    for i in six.moves.range(elem_size):
        res = [example[i] for example in batch]
        if i in indices_concat:
            res = _concat_arrays(res, padding[i])
        if i in indices_to_device:
            if i in indices_concat:
                res = to_device(device, res)
            else:
                res = [to_device(device, r) for r in res]
        result.append(res)

    return tuple(result)


class Transform(object):

    def __init__(self, light_head_rcnn):
        self.light_head_rcnn = light_head_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.light_head_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: LightHeadRCNN')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=1234)
    parser.add_argument('--batchsize', '-b', type=int, default=8)
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    # chainermn
    comm = chainermn.create_communicator('pure_nccl')
    device = comm.intra_rank

    np.random.seed(args.seed)
    random.seed(args.seed)

    # model
    light_head_rcnn = LightHeadRCNNResNet101(
        pretrained_model='imagenet',
        n_fg_class=len(coco_bbox_label_names))
    light_head_rcnn.use_preset('evaluate')
    model = LightHeadRCNNTrainChain(light_head_rcnn)
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # train dataset
    train_dataset = COCOBboxDataset(
        year='2017', split='train')

    # filter non-annotated data
    train_indices = np.array(
        [i for i, label in enumerate(train_dataset.slice[:, ['label']])
         if len(label[0]) > 0],
        dtype=np.int32)
    train_dataset = train_dataset.slice[train_indices]
    train_dataset = TransformDataset(
        train_dataset, ('img', 'bbox', 'label', 'scale'),
        Transform(model.light_head_rcnn))

    if comm.rank == 0:
        indices = np.arange(len(train_dataset))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train_dataset = train_dataset.slice[indices]
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=args.batchsize // comm.size)

    if comm.rank == 0:
        test_dataset = COCOBboxDataset(
            year='2017', split='val', use_crowded=True,
            return_crowded=True, return_area=True)
        test_iter = chainer.iterators.SerialIterator(
            test_dataset, batch_size=1, repeat=False, shuffle=False)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(momentum=0.9), comm)
    optimizer.setup(model)

    global_context_module = model.light_head_rcnn.head.global_context_module
    global_context_module.col_max.W.update_rule.add_hook(GradientScaling(3.0))
    global_context_module.col_max.b.update_rule.add_hook(GradientScaling(3.0))
    global_context_module.col.W.update_rule.add_hook(GradientScaling(3.0))
    global_context_module.col.b.update_rule.add_hook(GradientScaling(3.0))
    global_context_module.row_max.W.update_rule.add_hook(GradientScaling(3.0))
    global_context_module.row_max.b.update_rule.add_hook(GradientScaling(3.0))
    global_context_module.row.W.update_rule.add_hook(GradientScaling(3.0))
    global_context_module.row.b.update_rule.add_hook(GradientScaling(3.0))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

    model.light_head_rcnn.extractor.conv1.disable_update()
    model.light_head_rcnn.extractor.res2.disable_update()
    for link in model.links():
        if isinstance(link, L.BatchNormalization):
            link.disable_update()

    converter = functools.partial(
        concat_examples, padding=0,
        # img, bboxes, labels, scales
        indices_concat=[0, 2, 3],  # img, _, labels, scales
        indices_to_device=[0],     # img
    )

    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, converter=converter,
        device=device)
    trainer = chainer.training.Trainer(
        updater, (30, 'epoch'), out=args.out)

    @make_shift('lr')
    def lr_scheduler(trainer):
        base_lr = 0.0005 * 1.25 * args.batchsize
        warm_up_duration = 500
        warm_up_rate = 1 / 3

        iteration = trainer.updater.iteration
        epoch = trainer.updater.epoch
        if iteration < warm_up_duration:
            rate = warm_up_rate \
                + (1 - warm_up_rate) * iteration / warm_up_duration
        elif epoch < 19:
            rate = 1
        elif epoch < 25:
            rate = 0.1
        else:
            rate = 0.01
        return rate * base_lr

    trainer.extend(lr_scheduler)

    if comm.rank == 0:
        # interval
        log_interval = 100, 'iteration'
        plot_interval = 3000, 'iteration'
        print_interval = 20, 'iteration'

        # training extensions
        model_name = model.light_head_rcnn.__class__.__name__
        trainer.extend(
            chainer.training.extensions.snapshot_object(
                model.light_head_rcnn,
                filename='%s_model_iter_{.updater.iteration}.npz'
                         % model_name),
            trigger=(1, 'epoch'))
        trainer.extend(
            extensions.observe_lr(),
            trigger=log_interval)
        trainer.extend(
            extensions.LogReport(log_name='log.json', trigger=log_interval))
        report_items = [
            'iteration', 'epoch', 'elapsed_time', 'lr',
            'main/loss',
            'main/rpn_loc_loss',
            'main/rpn_cls_loss',
            'main/roi_loc_loss',
            'main/roi_cls_loss',
            'validation/main/map/iou=0.50:0.95/area=all/max_dets=100',
        ]
        trainer.extend(
            extensions.PrintReport(report_items), trigger=print_interval)
        trainer.extend(
            extensions.ProgressBar(update_interval=10))

        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(
                    ['main/loss'],
                    file_name='loss.png', trigger=plot_interval),
                trigger=plot_interval)

        trainer.extend(
            DetectionCOCOEvaluator(
                test_iter, model.light_head_rcnn,
                label_names=coco_bbox_label_names),
            trigger=ManualScheduleTrigger([19, 25], 'epoch'))
        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
