import fire
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

from chainercv.datasets import TransformDataset
from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVisReport
from chainercv import transforms

from chainercv.links import FasterRCNNResNet
from chainercv.links import FasterRCNNVGG


mean_pixel = np.array([102.9801, 115.9465, 122.7717])[:, None, None]


def main(gpu=-1, model='vgg', batch_size=1,
         iteration=100000, lr=1e-3, out='result', resume=''):
    train_data = VOCDetectionDataset(mode='trainval', year='2007')
    test_data = VOCDetectionDataset(mode='test', year='2007')

    def transform(in_data):
        img, bbox, label = in_data
        img -= mean_pixel 
        # Resize bounding box to a shape
        # with the smaller edge at least at length 600
        _, H, W = img.shape
        img = transforms.scale(img, 600)
        _, o_H, o_W = img.shape
        # Prevent the biggest axis from being more than MAX_SIZE
        if max(o_H, o_W) > 1200:
            rate = 1200 / float(max(o_H, o_W))
            img = transforms.resize(img, (int(o_W * rate), int(o_H * rate)))
            _, o_H, o_W = img.shape
        bbox = transforms.resize_bbox(bbox, (W, H), (o_W, o_H))

        # horizontally flip
        img, params = transforms.random_flip(img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(bbox, (o_W, o_H), params['x_flip'])
        return img, bbox, label, float(o_W) / float(W)

    train_data = TransformDataset(train_data, transform)
    test_data = TransformDataset(test_data, transform)

    if model == 'vgg':
        model = FasterRCNNVGG()
    elif model == 'resnet':
        model = FasterRCNNResNet()
    if gpu != -1:
        model.to_gpu(gpu)
        chainer.cuda.get_device(gpu).use()
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=3, shared_mem=100000000)

    # train_iter = chainer.iterators.SerialIterator(train_data, batch_size=1)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out)

    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(50000, 'iteration'))

    log_interval = 20, 'iteration'
    val_interval = 5000, 'iteration'
    plot_interval = 100, 'iteration'
    trainer.extend(
        chainer.training.extensions.observe_lr(),
        trigger=log_interval
    )
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/loss_bbox',
         'main/loss_cls',
         'main/rpn_loss_cls',
         'main/rpn_loss_bbox',
         ]), trigger=plot_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # visualize training
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/rpn_loss_cls'],
                file_name='rpn_loss_cls.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/rpn_loss_bbox'],
                file_name='rpn_loss_bbox.png'
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/loss_cls'],
                file_name='loss_cls.png'
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/loss_bbox'],
                file_name='loss_bbox.png'
            ),
            trigger=plot_interval
        )

    def vis_transform(in_data):
        img, bbox, label, scale = in_data
        img += mean_pixel
        img = transforms.chw_to_pil_image(img)
        return img, bbox, label

    # trainer.extend(
    #     DetectionVisReport(
    #         range(10),  # visualize outputs for the first 10 data of train_data
    #         train_data,
    #         model,
    #         filename_base='detection_train',
    #         predict_func=model.predict_bbox,
    #         vis_transform=vis_transform
    #     ),
    #     trigger=val_interval, invoke_before_training=True
    # )
    # trainer.extend(
    #     DetectionVisReport(
    #         range(10),  # visualize outputs for the first 10 data of 
    #         test_data,
    #         model,
    #         filename_base='detection_test',
    #         predict_func=model.predict_bbox,
    #         vis_transform=vis_transform
    #     ),
    #     trigger=val_interval, invoke_before_training=True
    # )

    trainer.extend(extensions.dump_graph('main/loss'))

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()


if __name__ == '__main__':
    fire.Fire(main)
