import argparse
import multiprocessing
import numpy as np

import chainer
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.training.extensions import PolynomialShift

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.datasets import VOCSemanticSegmentationDataset

from chainercv.links import DeepLabV3plusXception65

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv import transforms

import PIL

import chainermn

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


class Transform(object):

    def __init__(
            self, crop_size, mean, scale_range=[0.5, 2.0], resize=None):
        self.crop_size = crop_size
        self.mean = mean
        self.scale_range = scale_range
        self.resize = resize

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape

        # 1. resize longer side to 513 when ADE20k
        if self.resize is not None:
            H, W = self.resize, self.resize
            img = transforms.resize(img, (H, W), PIL.Image.BICUBIC)
            label = transforms.resize(
                label[None], (H, W), PIL.Image.NEAREST)[0]

        # 2. random scale
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        H, W = int(scale * H), int(scale * W)
        img = transforms.resize(img, (H, W), PIL.Image.BICUBIC)
        label = transforms.resize(
            label[None], (H, W), PIL.Image.NEAREST)[0]
            
        # 3. pad to bigger than or equal to crop size with mean pixel
        h = max(self.crop_size[0], H)
        w = max(self.crop_size[1], W)
        _img, _label = img, label
        img = np.empty((3, h, w), dtype=np.float32)
        label = np.empty((h, w), dtype=np.int32)
        img[:] = self.mean
        label[:] = -1
        img[:, :H, :W] = _img
        label[:H, :W] = _label

        # 4. random crop
        if (h, w) != self.crop_size:
            img, param = transforms.random_crop(img, self.crop_size, True)
            label = label[param['y_slice'], param['x_slice']]

        # 5. random hirizontal flip
        if np.random.rand() > 0.5:
            img = transforms.flip(img, x_flip=True)
            label = transforms.flip(label[None], x_flip=True)[0]

        # 6. scale values to [-1.0, 1.0]
        img = img / 127.5 - 1.0

        return img, label


class TrainChain(chainer.Chain):

    def __init__(self, model, crop_size):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model
            self.crop_size = crop_size

    def __call__(self, imgs, labels):
        h = self.model(imgs)
        h = F.resize_images(h, self.crop_size)
        loss = F.softmax_cross_entropy(h, labels)

        chainer.reporter.report({'loss': loss}, self)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='auto')
    parser.add_argument('--dataset',
                        choices=('ade20k', 'cityscapes', 'voc'))
    parser.add_argument('--lr', default=0.007, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--out', default='results')
    parser.add_argument('--iteration', default=None, type=int)
    parser.add_argument('--communicator', default='hierarchical')
    args = parser.parse_args()

    dataset_cfgs = {
        'ade20k': {
            'input_size': (473, 473),
            'label_names': ade20k_semantic_segmentation_label_names,
            'iteration': 150000},
        'cityscapes': {
            'input_size': (769, 769),
            'label_names': cityscapes_semantic_segmentation_label_names,
            'iteration': 90000},
        'voc': {
            'input_size': (513, 513),
            'label_names': voc_semantic_segmentation_label_names,
            'iteration': 30000},
    }
    dataset_cfg = dataset_cfgs[args.dataset]

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    extractor_kwargs = {'bn_kwargs': {'comm': comm, 'decay': 0.9997}}
    aspp_kwargs = {'bn_kwargs': {'comm': comm, 'decay': 0.9997}}
    decoder_kwargs = {'bn_kwargs': {'comm': comm, 'decay': 0.9997}}
    model = DeepLabV3plusXception65(
        n_class=len(dataset_cfg['label_names']),
        min_input_size=dataset_cfg['input_size'],
        scales=(1.0,), flip=False, extractor_kwargs=extractor_kwargs,
        aspp_kwargs=aspp_kwargs, decoder_kwargs=decoder_kwargs)
    train_chain = TrainChain(model, dataset_cfg['input_size'])

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        train_chain.to_gpu()

    if args.iteration is None:
        n_iter = dataset_cfg['iteration']
    else:
        n_iter = args.iteration

    if args.dataset == 'ade20k':
        train = ADE20KSemanticSegmentationDataset(
            data_dir=args.data_dir, split='train')
        if comm.rank == 0:
            val = ADE20KSemanticSegmentationDataset(
                data_dir=args.data_dir, split='val')
        label_names = ade20k_semantic_segmentation_label_names
    elif args.dataset == 'cityscapes':
        train = CityscapesSemanticSegmentationDataset(
            args.data_dir,
            label_resolution='fine', split='train')
        if comm.rank == 0:
            val = CityscapesSemanticSegmentationDataset(
                args.data_dir,
                label_resolution='fine', split='val')
        label_names = cityscapes_semantic_segmentation_label_names
    elif args.dataset == 'voc':
        train = VOCSemanticSegmentationDataset(
            args.data_dir, split='train')
        if comm.rank == 0:
            val = VOCSemanticSegmentationDataset(
                args.data_dir, split='val')
        label_names = voc_semantic_segmentation_label_names
    train = TransformDataset(
        train,
        ('img', 'label'),
        Transform(dataset_cfg['input_size'],
                  model.feature_extractor.mean))

    if comm.rank == 0:
        indices = np.arange(len(train))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train = train.slice[indices]

    train_iter = chainer.iterators.MultiprocessIterator(
        train, batch_size=args.batch_size, n_processes=1)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(args.lr, 0.9), comm)
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name not in ('beta', 'gamma'):
            param.update_rule.add_hook(chainer.optimizer.WeightDecay(4e-5))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (n_iter, 'iteration'), args.out)
    trainer.extend(
        PolynomialShift('lr', 0.9, n_iter, optimizer=optimizer),
        trigger=(1, 'iteration'))

    log_interval = 10, 'iteration'

    if comm.rank == 0:
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'elapsed_time', 'lr', 'main/loss',
             'validation/main/miou', 'validation/main/mean_class_accuracy',
             'validation/main/pixel_accuracy']),
            trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(
            extensions.snapshot_object(
                train_chain.model, 'snapshot_model_{.updater.iteration}.npz'),
            trigger=(n_iter, 'iteration'))
        val_iter = chainer.iterators.SerialIterator(
            val, batch_size=1, repeat=False, shuffle=False)
        trainer.extend(
            SemanticSegmentationEvaluator(
                val_iter, model,
                label_names),
            trigger=(n_iter, 'iteration'))

    trainer.run()


if __name__ == '__main__':
    main()
