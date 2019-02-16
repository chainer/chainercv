from __future__ import division

import argparse
import multiprocessing
import numpy as np

import chainer
import chainer.links as L
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions

import chainermn

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.chainer_experimental.training.extensions import make_shift
from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv import transforms

from chainercv.links.model.fpn import head_loss_post
from chainercv.links.model.fpn import head_loss_pre
from chainercv.links.model.fpn import rpn_loss

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


class TrainChain(chainer.Chain):

    def __init__(self, model):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, imgs, bboxes, labels):
        x, scales = self.model.prepare(imgs)
        bboxes = [self.xp.array(bbox) * scale
                  for bbox, scale in zip(bboxes, scales)]
        labels = [self.xp.array(label) for label in labels]

        with chainer.using_config('train', False):
            hs = self.model.extractor(x)

        rpn_locs, rpn_confs = self.model.rpn(hs)
        anchors = self.model.rpn.anchors(h.shape[2:] for h in hs)
        rpn_loc_loss, rpn_conf_loss = rpn_loss(
            rpn_locs, rpn_confs, anchors,
            [(int(img.shape[1] * scale), int(img.shape[2] * scale))
             for img, scale in zip(imgs, scales)],
            bboxes)

        rois, roi_indices = self.model.rpn.decode(
            rpn_locs, rpn_confs, anchors, x.shape)
        rois = self.xp.vstack([rois] + bboxes)
        roi_indices = self.xp.hstack(
            [roi_indices]
            + [self.xp.array((i,) * len(bbox))
               for i, bbox in enumerate(bboxes)])
        rois, roi_indices = self.model.head.distribute(rois, roi_indices)
        rois, roi_indices, head_gt_locs, head_gt_labels = head_loss_pre(
            rois, roi_indices, self.model.head.std, bboxes, labels)
        head_locs, head_confs = self.model.head(hs, rois, roi_indices)
        head_loc_loss, head_conf_loss = head_loss_post(
            head_locs, head_confs,
            roi_indices, head_gt_locs, head_gt_labels, len(x))

        loss = rpn_loc_loss + rpn_conf_loss + head_loc_loss + head_conf_loss
        chainer.reporter.report({
            'loss': loss,
            'loss/rpn/loc': rpn_loc_loss, 'loss/rpn/conf': rpn_conf_loss,
            'loss/head/loc': head_loc_loss, 'loss/head/conf': head_conf_loss},
            self)

        return loss


def transform(in_data):
    img, bbox, label = in_data

    img, params = transforms.random_flip(
        img, x_random=True, return_param=True)
    bbox = transforms.flip_bbox(
        bbox, img.shape[1:], x_flip=params['x_flip'])

    return img, bbox, label


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=('faster_rcnn_fpn_resnet50', 'faster_rcnn_fpn_resnet101'),
        default='faster_rcnn_fpn_resnet50')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--iteration', type=int, default=90000)
    parser.add_argument('--step', type=int, nargs='*', default=[60000, 80000])
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    if args.model == 'faster_rcnn_fpn_resnet50':
        model = FasterRCNNFPNResNet50(
            n_fg_class=len(coco_bbox_label_names), pretrained_model='imagenet')
    elif args.model == 'faster_rcnn_fpn_resnet101':
        model = FasterRCNNFPNResNet101(
            n_fg_class=len(coco_bbox_label_names), pretrained_model='imagenet')

    model.use_preset('evaluate')
    train_chain = TrainChain(model)
    chainer.cuda.get_device_from_id(device).use()
    train_chain.to_gpu()

    train = TransformDataset(
        COCOBboxDataset(year='2017', split='train'),
        ('img', 'bbox', 'label'), transform)

    if comm.rank == 0:
        indices = np.arange(len(train))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train = train.slice[indices]

    train_iter = chainer.iterators.MultithreadIterator(
        train, args.batchsize // comm.size)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(), comm)
    optimizer.setup(train_chain)
    optimizer.add_hook(WeightDecay(0.0001))

    model.extractor.base.conv1.disable_update()
    model.extractor.base.res2.disable_update()
    for link in model.links():
        if isinstance(link, L.BatchNormalization):
            link.disable_update()

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(
        updater, (args.iteration * 16 / args.batchsize, 'iteration'), args.out)

    @make_shift('lr')
    def lr_schedule(trainer):
        base_lr = 0.02 * args.batchsize / 16
        warm_up_duration = 500
        warm_up_rate = 1 / 3

        iteration = trainer.updater.iteration
        if iteration < warm_up_duration:
            rate = warm_up_rate \
                + (1 - warm_up_rate) * iteration / warm_up_duration
        else:
            rate = 1
            for step in args.step:
                if iteration >= step * 16 / args.batchsize:
                    rate *= 0.1

        return base_lr * rate

    trainer.extend(lr_schedule)

    if comm.rank == 0:
        log_interval = 10, 'iteration'
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'lr', 'main/loss',
             'main/loss/rpn/loc', 'main/loss/rpn/conf',
             'main/loss/head/loc', 'main/loss/head/conf']),
            trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
        trainer.extend(
            extensions.snapshot_object(
                model, 'model_iter_{.updater.iteration}'),
            trigger=(90000 * 16 / args.batchsize, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer, strict=False)

    trainer.run()


if __name__ == '__main__':
    main()
