import argparse
import copy
import multiprocessing
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training.extensions import PolynomialShift

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset

from chainercv.experimental.links import PSPNetResNet101
from chainercv.experimental.links import PSPNetResNet50

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.links import Conv2DBNActiv
from chainercv import transforms

from chainercv.links.model.ssd import GradientScaling

import PIL

import chainermn

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


def create_mnbn_model(link, comm):
    """Returns a copy of a model with BN replaced by Multi-node BN."""
    if isinstance(link, chainer.links.BatchNormalization):
        mnbn = chainermn.links.MultiNodeBatchNormalization(
            size=link.avg_mean.shape,
            comm=comm,
            decay=link.decay,
            eps=link.eps,
            dtype=link.avg_mean.dtype,
            use_gamma=hasattr(link, 'gamma'),
            use_beta=hasattr(link, 'beta'),
        )
        mnbn.copyparams(link)
        for name in link._persistent:
            mnbn.__dict__[name] = copy.deepcopy(link.__dict__[name])
        return mnbn
    elif isinstance(link, chainer.Chain):
        new_children = [
            (child_name, create_mnbn_model(
                link.__dict__[child_name], comm))
            for child_name in link._children
        ]
        new_link = copy.deepcopy(link)
        for name, new_child in new_children:
            new_link.__dict__[name] = new_child
        return new_link
    elif isinstance(link, chainer.ChainList):
        new_children = [
            create_mnbn_model(l, comm) for l in link]
        new_link = copy.deepcopy(link)
        for i, new_child in enumerate(new_children):
            new_link._children[i] = new_child
        return new_link
    else:
        return copy.deepcopy(link)


class Transform(object):

    def __init__(
            self, mean,
            crop_size, scale_range=[0.5, 2.0]):
        self.mean = mean
        self.scale_range = scale_range
        self.crop_size = crop_size

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Scale
        scaled_H = int(scale * H)
        scaled_W = int(scale * W)
        img = transforms.resize(img, (scaled_H, scaled_W), PIL.Image.BICUBIC)
        label = transforms.resize(
            label[None], (scaled_H, scaled_W), PIL.Image.NEAREST)[0]

        # Crop
        if (scaled_H < self.crop_size[0]) or (scaled_W < self.crop_size[1]):
            shorter_side = min(img.shape[1:])
            img, param = transforms.random_crop(
                img, (shorter_side, shorter_side), True)
        else:
            img, param = transforms.random_crop(img, self.crop_size, True)
        label = label[param['y_slice'], param['x_slice']]

        # Rotate
        angle = np.random.uniform(-10, 10)
        img = transforms.rotate(img, angle, expand=False)
        label = transforms.rotate(
            label[None], angle, expand=False,
            interpolation=PIL.Image.NEAREST,
            fill=-1)[0]

        # Resize
        if ((img.shape[1] < self.crop_size[0])
                or (img.shape[2] < self.crop_size[1])):
            img = transforms.resize(img, self.crop_size, PIL.Image.BICUBIC)
        if ((label.shape[0] < self.crop_size[0])
                or (label.shape[1] < self.crop_size[1])):
            label = transforms.resize(
                label[None].astype(np.float32),
                self.crop_size, PIL.Image.NEAREST)
            label = label.astype(np.int32)[0]
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = transforms.flip(img, x_flip=True)
            label = transforms.flip(label[None], x_flip=True)[0]

        # Mean subtraction
        img = img - self.mean
        return img, label


class TrainChain(chainer.Chain):

    def __init__(self, model):
        initialW = chainer.initializers.HeNormal()
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model
            self.aux_conv1 = Conv2DBNActiv(
                None, 512, 3, 1, 1, initialW=initialW)
            self.aux_conv2 = L.Convolution2D(
                None, model.n_class, 3, 1, 1, False, initialW=initialW)

    def __call__(self, imgs, labels):
        h_aux, h_main = self.model.extractor(imgs)
        h_aux = F.dropout(self.aux_conv1(h_aux), ratio=0.1)
        h_aux = self.aux_conv2(h_aux)
        h_aux = F.resize_images(h_aux, imgs.shape[2:])

        h_main = self.model.ppm(h_main)
        h_main = F.dropout(self.model.head_conv1(h_main), ratio=0.1)
        h_main = self.model.head_conv2(h_main)
        h_main = F.resize_images(h_main, imgs.shape[2:])

        aux_loss = F.softmax_cross_entropy(h_aux, labels)
        main_loss = F.softmax_cross_entropy(h_main, labels)
        loss = 0.4 * aux_loss + main_loss

        chainer.reporter.report({'loss': loss}, self)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='auto')
    parser.add_argument('--dataset',
                        choices=('ade20k', 'cityscapes'))
    parser.add_argument('--model',
                        choices=('pspnet_resnet101', 'pspnet_resnet50'))
    parser.add_argument('--lr', default=1e-2)
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--out', default='result')
    parser.add_argument('--iteration', default=None, type=int)
    parser.add_argument('--communicator', default='hierarchical')
    args = parser.parse_args()

    dataset_cfgs = {
        'ade20k': {
            'input_size': (473, 473),
            'label_names': ade20k_semantic_segmentation_label_names,
            'iteration': 150000},
        'cityscapes': {
            'input_size': (713, 713),
            'label_names': cityscapes_semantic_segmentation_label_names,
            'iteration': 90000}
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

    n_class = len(dataset_cfg['label_names'])
    if args.model == 'pspnet_resnet101':
        model = PSPNetResNet101(
            n_class, pretrained_model='imagenet',
            input_size=dataset_cfg['input_size'])
    elif args.model == 'pspnet_resnet50':
        model = PSPNetResNet50(
            n_class, pretrained_model='imagenet',
            input_size=dataset_cfg['input_size'])
    train_chain = create_mnbn_model(TrainChain(model), comm)
    model = train_chain.model
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
    train = TransformDataset(
        train,
        ('img', 'label'),
        Transform(model.mean, dataset_cfg['input_size']))

    if comm.rank == 0:
        indices = np.arange(len(train))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train = train.slice[indices]

    train_iter = chainer.iterators.MultiprocessIterator(
        train, batch_size=args.batchsize, n_processes=2)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(args.lr, 0.9), comm)
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name not in ('beta', 'gamma'):
            param.update_rule.add_hook(chainer.optimizer.WeightDecay(1e-4))
    for l in [
            model.ppm, model.head_conv1, model.head_conv2,
            train_chain.aux_conv1, train_chain.aux_conv2]:
        for param in l.params():
            param.update_rule.add_hook(GradientScaling(10))

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
