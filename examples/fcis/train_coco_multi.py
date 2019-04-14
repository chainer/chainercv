from __future__ import division

import argparse
import multiprocessing
import numpy as np

import chainer
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
import chainermn

from chainercv.chainer_experimental.datasets.sliceable \
    import ConcatenatedDataset
from chainercv.chainer_experimental.datasets.sliceable \
    import TransformDataset
from chainercv.chainer_experimental.training.extensions import make_shift
from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.datasets import COCOInstanceSegmentationDataset
from chainercv.experimental.links import FCISResNet101
from chainercv.experimental.links import FCISTrainChain
from chainercv.experimental.links.model.fcis.utils.proposal_target_creator \
    import ProposalTargetCreator
from chainercv.extensions import InstanceSegmentationCOCOEvaluator
from chainercv.links.model.ssd import GradientScaling

from train_sbd import concat_examples
from train_sbd import Transform


# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: FCIS')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument(
        '--lr', '-l', type=float, default=None,
        help='Learning rate for multi GPUs')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--epoch', '-e', type=int, default=18)
    parser.add_argument('--cooldown-epoch', '-ce', type=int, default=12)
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    # chainermn
    comm = chainermn.create_communicator()
    device = comm.intra_rank

    np.random.seed(args.seed)

    # model
    proposal_creator_params = FCISResNet101.proposal_creator_params
    proposal_creator_params['min_size'] = 2
    fcis = FCISResNet101(
        n_fg_class=len(coco_instance_segmentation_label_names),
        anchor_scales=(4, 8, 16, 32),
        pretrained_model='imagenet', iter2=False,
        proposal_creator_params=proposal_creator_params)
    fcis.use_preset('coco_evaluate')
    proposal_target_creator = ProposalTargetCreator()
    proposal_target_creator.neg_iou_thresh_lo = 0.0
    model = FCISTrainChain(
        fcis, proposal_target_creator=proposal_target_creator)

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # train dataset
    train_dataset = COCOInstanceSegmentationDataset(
        year='2014', split='train')
    vmml_dataset = COCOInstanceSegmentationDataset(
        year='2014', split='valminusminival')

    # filter non-annotated data
    train_indices = np.array(
        [i for i, label in enumerate(train_dataset.slice[:, ['label']])
         if len(label[0]) > 0],
        dtype=np.int32)
    train_dataset = train_dataset.slice[train_indices]
    vmml_indices = np.array(
        [i for i, label in enumerate(vmml_dataset.slice[:, ['label']])
         if len(label[0]) > 0],
        dtype=np.int32)
    vmml_dataset = vmml_dataset.slice[vmml_indices]

    train_dataset = TransformDataset(
        ConcatenatedDataset(train_dataset, vmml_dataset),
        ('img', 'mask', 'label', 'bbox', 'scale'),
        Transform(model.fcis))
    if comm.rank == 0:
        indices = np.arange(len(train_dataset))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train_dataset = train_dataset.slice[indices]
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=args.batchsize // comm.size)

    # test dataset
    if comm.rank == 0:
        test_dataset = COCOInstanceSegmentationDataset(
            year='2014', split='minival', use_crowded=True,
            return_crowded=True, return_area=True)
        indices = np.arange(len(test_dataset))
        test_dataset = test_dataset.slice[indices]
        test_iter = chainer.iterators.SerialIterator(
            test_dataset, batch_size=1, repeat=False, shuffle=False)

    # optimizer
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(momentum=0.9),
        comm)
    optimizer.setup(model)

    model.fcis.head.conv1.W.update_rule.add_hook(GradientScaling(3.0))
    model.fcis.head.conv1.b.update_rule.add_hook(GradientScaling(3.0))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    for param in model.params():
        if param.name in ['beta', 'gamma']:
            param.update_rule.enabled = False
    model.fcis.extractor.conv1.disable_update()
    model.fcis.extractor.res2.disable_update()

    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, converter=concat_examples,
        device=device)

    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    # lr scheduler
    @make_shift('lr')
    def lr_scheduler(trainer):
        if args.lr is None:
            base_lr = 0.0005 * args.batchsize
        else:
            base_lr = args.lr

        iteration = trainer.updater.iteration
        epoch = trainer.updater.epoch
        if (iteration * comm.size) < 2000:
            rate = 0.1
        elif epoch < args.cooldown_epoch:
            rate = 1
        else:
            rate = 0.1
        return rate * base_lr

    trainer.extend(lr_scheduler)

    if comm.rank == 0:
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
        report_items = [
            'iteration', 'epoch', 'elapsed_time', 'lr',
            'main/loss',
            'main/rpn_loc_loss',
            'main/rpn_cls_loss',
            'main/roi_loc_loss',
            'main/roi_cls_loss',
            'main/roi_mask_loss',
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
            InstanceSegmentationCOCOEvaluator(
                test_iter, model.fcis,
                label_names=coco_instance_segmentation_label_names),
            trigger=ManualScheduleTrigger(
                [len(train_dataset) * args.cooldown_epoch,
                 len(train_dataset) * args.epoch], 'iteration'))

        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
