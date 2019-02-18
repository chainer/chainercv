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

    def prepare_mask(self, masks, resized_sizes, pad_size):
        resized_masks = []
        for size, mask in zip(resized_sizes, masks):
            resized_masks.append(transforms.resize(
                mask.astype(np.float32),
                size, interpolation=PIL.Image.NEAREST).astype(np.bool))
        pad_masks = []
        for mask in resized_masks:
            n_class, H, W = mask.shape
            pad_mask = self.xp.zeros(
                (n_class, pad_size[0], pad_size[1]), dtype=np.bool)
            pad_mask[:, :H, :W] = self.xp.array(mask)
            pad_masks.append(pad_mask)
        return pad_masks

    def __call__(self, imgs, masks, labels, bboxes):
        x, scales, resized_sizes = self.model.prepare(imgs)
        B, _, pad_H, pad_W = x.shape
        masks = self.prepare_mask(masks, resized_sizes, (pad_H, pad_W))
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
        return loss


def transform(in_data):
    img, mask, label, bbox = in_data

    img, params = transforms.random_flip(
        img, x_random=True, return_param=True)
    mask = transforms.flip(mask, x_flip=params['x_flip'])
    bbox = transforms.flip_bbox(
        bbox, img.shape[1:], x_flip=params['x_flip'])

    return img, mask, label, bbox


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
            split='train', return_bbox=True),
        ('img', 'mask', 'label', 'bbox'), transform)

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

    trainer.run()


if __name__ == '__main__':
    main()
