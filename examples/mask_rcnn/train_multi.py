import argparse
import multiprocessing
import numpy as np
import PIL

import chainer
import chainer.links as L
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions

import chainermn

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.chainer_experimental.training.extensions import make_shift
from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.datasets import COCOInstanceSegmentationDataset
from chainercv.links import MaskRCNNFPNResNet101
from chainercv.links import MaskRCNNFPNResNet50
from chainercv import transforms

from chainercv.links.model.fpn import head_loss_post
from chainercv.links.model.fpn import head_loss_pre
from chainercv.links.model.fpn import rpn_loss
from chainercv.links.model.mask_rcnn import mask_loss_post
from chainercv.links.model.mask_rcnn import mask_loss_pre

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
# try:
#     import cv2
#     cv2.setNumThreads(0)
# except ImportError:
#     pass


class TrainChain(chainer.Chain):

    def __init__(self, model):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, imgs, masks, labels, bboxes):
        B = len(imgs)
        pad_size = np.array(
            [im.shape[1:] for im in imgs]).max(axis=0)
        pad_size = (
            np.ceil(pad_size / self.model.stride) * self.model.stride).astype(int)
        x = np.zeros(
            (len(imgs), 3, pad_size[0], pad_size[1]), dtype=np.float32)
        for i, img in enumerate(imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img
        x = self.xp.array(x)

        pad_masks = [
            self.xp.zeros(
                (mask.shape[0], pad_size[0], pad_size[1]), dtype=np.bool)
            for mask in masks]
        for i, mask in enumerate(masks):
            _, H, W = mask.shape
            pad_masks[i][:, :H, :W] = self.xp.array(mask)
        masks = pad_masks

        bboxes = [self.xp.array(bbox) for bbox in bboxes]
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

        mask_rois, mask_roi_indices, gt_segms, gt_mask_labels = mask_loss_pre(
            rois, roi_indices, masks, head_gt_labels,
            self.model.mask_head.mask_size)
        n_roi = sum([len(roi) for roi in mask_rois])
        if n_roi > 0:
            segms = self.model.mask_head(hs, mask_rois, mask_roi_indices)
            mask_loss = mask_loss_post(
                segms, mask_roi_indices, gt_segms, gt_mask_labels, B)
            loss = (rpn_loc_loss + rpn_conf_loss + 
                head_loc_loss + head_conf_loss + mask_loss)
            chainer.reporter.report({
                'loss': loss,
                'loss/rpn/loc': rpn_loc_loss, 'loss/rpn/conf': rpn_conf_loss,
                'loss/head/loc': head_loc_loss, 'loss/head/conf': head_conf_loss,
                'loss/mask': mask_loss},
                self)
        else:
            # ChainerMN hangs when a subset of nodes has a different
            # computational graph from the rest.
            loss = chainer.Variable(self.xp.array(0, dtype=np.float32))
            self.zerograds()
        return loss


class Transform(object):

    def __init__(self, mean, min_size, max_size):
        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        import time
        start = time.time()
        img, mask, label, bbox = in_data
        original = mask.shape
        # Flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        mask = transforms.flip(mask, x_flip=params['x_flip'])
        bbox = transforms.flip_bbox(
            bbox, img.shape[1:], x_flip=params['x_flip'])

        # TODO: make this part reusable
        # Scaling
        _, H, W = img.shape
        scale = self.min_size / min(H, W)
        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
        H, W = int(H * scale), int(W * scale)
        img = transforms.resize(img, (H, W))
        mask = transforms.resize(
            mask.astype(np.float32),
            (H, W), interpolation=PIL.Image.NEAREST).astype(np.bool)
        bbox = bbox * scale

        # Subtract mean
        img -= self.mean
        return img, mask, label, bbox, scale


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


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
    parser.add_argument('--cprofile', action='store_true', help='cprofile')
    args = parser.parse_args()
    # chainer.global_config.cv_resize_backend = 'PIL'
    # chainer.global_config.cv_read_image_backend = 'PIL'

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank
    global rank
    rank = comm.rank

    if args.model == 'mask_rcnn_fpn_resnet50':
        model = MaskRCNNFPNResNet50(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model='imagenet')
    elif args.model == 'mask_rcnn_fpn_resnet101':
        model = MaskRCNNFPNResNet101(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    train_chain = TrainChain(model)
    chainer.cuda.get_device_from_id(device).use()
    train_chain.to_gpu()

    train = TransformDataset(
        COCOInstanceSegmentationDataset(
            data_dir='/home/yuyu2172/coco',
            split='train', return_bbox=True),
        ('img', 'mask', 'label', 'bbox'),
        Transform(model.extractor.mean, model.min_size, model.max_size))

    if comm.rank == 0:
        indices = np.arange(len(train))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train = train.slice[indices]

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize // comm.size,
        n_processes=args.batchsize // comm.size, shared_mem=100 * 1000 * 1000 * 4)

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
             'main/loss/mask'
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

    if args.cprofile:
        import cProfile
        import io
        import pstats
        print('cprofiling')
        pr = cProfile.Profile()
        pr.enable()
    trainer.run()
    if args.cprofile:
        pr.disable()
        s = io.StringIO()
        sort_by = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.print_stats()
        if comm.rank == 0:
            print(s.getvalue())

        pr.dump_stats('{0}/rank_{1}.cprofile'.format(args.out, comm.rank))


if __name__ == '__main__':
    main()
