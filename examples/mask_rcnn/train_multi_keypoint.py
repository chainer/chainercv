import argparse
import multiprocessing
import numpy as np
import random

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions

import chainermn

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.chainer_experimental.training.extensions import make_shift
from chainercv.datasets import COCOKeypointDataset
from chainercv.links import MaskRCNNFPNResNet101
from chainercv.links import MaskRCNNFPNResNet50
from chainercv.links.model.mask_rcnn.misc import scale_img
from chainercv import transforms

from chainercv.links.model.fpn import head_loss_post
from chainercv.links.model.fpn import head_loss_pre
from chainercv.links.model.fpn import rpn_loss
from chainercv.links.model.mask_rcnn import keypoint_loss_pre
from chainercv.links.model.mask_rcnn import keypoint_loss_post

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

    def __call__(self, imgs, points, visibles, labels, bboxes):
        B = len(imgs)
        pad_size = np.array(
            [im.shape[1:] for im in imgs]).max(axis=0)
        pad_size = (
            np.ceil(
                pad_size / self.model.stride) * self.model.stride).astype(int)
        x = np.zeros(
            (len(imgs), 3, pad_size[0], pad_size[1]), dtype=np.float32)
        for i, img in enumerate(imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img
        x = self.xp.array(x)

        points = [self.xp.array(point) for point in points]
        visibles = [self.xp.array(visible) for visible in visibles]

        bboxes = [self.xp.array(bbox) for bbox in bboxes]
        assert all([np.all(label == 0) for label in labels])
        labels = [self.xp.array(label) for label in labels]
        sizes = [img.shape[1:] for img in imgs]

        with chainer.using_config('train', False):
            hs = self.model.extractor(x)

        rpn_locs, rpn_confs = self.model.rpn(hs)
        anchors = self.model.rpn.anchors(h.shape[2:] for h in hs)
        rpn_loc_loss, rpn_conf_loss = rpn_loss(
            rpn_locs, rpn_confs, anchors, sizes, bboxes)

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
            roi_indices, head_gt_locs, head_gt_labels, B)
        losses = [
            rpn_loc_loss + rpn_conf_loss + head_loc_loss + head_conf_loss]

        point_rois, point_roi_indices, gt_head_points, gt_head_visibles = keypoint_loss_pre(
            rois, roi_indices, points, visibles, bboxes, head_gt_labels,
            self.model.keypoint_head.point_map_size)
        n_roi = sum([len(roi) for roi in point_rois])
        if n_roi > 0:
            point_maps = self.model.keypoint_head(hs, point_rois, point_roi_indices)
            point_loss = keypoint_loss_post(
                point_maps, point_roi_indices,
                gt_head_points, gt_head_visibles, B)
        else:
            # Compute dummy variables to complete the computational graph
            point_rois[0] = self.xp.array([[0, 0, 1, 1]], dtype=np.float32)
            point_roi_indices[0] = self.xp.array([0], dtype=np.int32)
            point_maps = self.model.keypoint_head(hs, point_rois, point_roi_indices)
            point_loss = 0 * F.sum(point_maps)
        losses.append(point_loss)
        loss = sum(losses)
        chainer.reporter.report({
            'loss': loss,
            'loss/rpn/loc': rpn_loc_loss, 'loss/rpn/conf': rpn_conf_loss,
            'loss/head/loc': head_loc_loss, 'loss/head/conf': head_conf_loss,
            'loss/point': point_loss},
            self)
        return loss


class Transform(object):

    def __init__(self, min_size, max_size, mean):
        if isinstance(min_size, (tuple, list)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.mean = mean

    def __call__(self, in_data):
        img, point, visible, label, bbox = in_data
        # Flipping
        size = img.shape[1:]
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        point = transforms.flip_point(
            point, size, x_flip=params['x_flip'])
        bbox = transforms.flip_bbox(
            bbox, size, x_flip=params['x_flip'])

        # Scaling and mean subtraction
        min_size = random.choice(self.min_size)
        img, scale = scale_img(img, min_size, self.max_size)
        img -= self.mean
        point = transforms.resize_point(point, size, img.shape[1:])
        bbox = bbox * scale
        return img, point, visible, label, bbox


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


def valid_annotation(visible):
    if len(visible) == 0:
        return False
    min_keypoint_per_image = 10
    n_visible = visible.sum()
    return n_visible >= min_keypoint_per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=('mask_rcnn_fpn_resnet50', 'mask_rcnn_fpn_resnet101'),
        default='mask_rcnn_fpn_resnet50')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--iteration', type=int, default=90000)
    parser.add_argument('--step', type=int, nargs='*', default=[60000, 80000])
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    parser.add_argument('--communicator', default='hierarchical')
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    if args.model == 'mask_rcnn_fpn_resnet50':
        model = MaskRCNNFPNResNet50(
            n_fg_class=1,
            pretrained_model='imagenet',
            mode='keypoint'
        )
    elif args.model == 'mask_rcnn_fpn_resnet101':
        model = MaskRCNNFPNResNet101(
            n_fg_class=1,
            pretrained_model='imagenet',
            mode='keypoint'
        )

    model.use_preset('evaluate')
    train_chain = TrainChain(model)
    chainer.cuda.get_device_from_id(device).use()
    train_chain.to_gpu()

    train = COCOKeypointDataset(split='train')
    indices = [i for i, visible in enumerate(train.slice[:, 'visible'])
               if valid_annotation(visible)]
    train = train.slice[indices]
    train = TransformDataset(
        train, ('img', 'point', 'visible', 'label', 'bbox'),
        Transform(
            (640, 672, 704, 736, 768, 800), model.max_size,
            model.extractor.mean))

    if comm.rank == 0:
        indices = np.arange(len(train))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train = train.slice[indices]

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize // comm.size,
        n_processes=args.batchsize // comm.size,
        shared_mem=3 * 1000 * 1000 * 4)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(), comm)
    optimizer.setup(train_chain)
    optimizer.add_hook(WeightDecay(0.0001))

    model.extractor.base.conv1.disable_update()
    model.extractor.base.res2.disable_update()
    for link in model.links():
        if isinstance(link, L.BatchNormalization):
            link.disable_update()

    n_iteration = args.iteration * 16 / args.batchsize
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(
        updater, (n_iteration, 'iteration'), args.out)

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
             'main/loss/head/loc', 'main/loss/head/conf',
             'main/loss/keypoint'
             ]),
            trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
        trainer.extend(
            extensions.snapshot_object(
                model, 'model_iter_{.updater.iteration}'),
            trigger=(n_iteration, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer, strict=False)

    trainer.run()


if __name__ == '__main__':
    main()
