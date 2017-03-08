import fire
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVisReport
from chainercv import transforms

from faster_rcnn import FasterRCNN
from updater import ParallelUpdater


def _shape_soft_min_hard_max(img_shape, soft_min, hard_max):
    lengths = np.array(img_shape).astype(np.float)
    min_length = np.min(lengths)
    scale = float(soft_min) / min_length
    lengths *= scale

    max_length = np.max(lengths)
    if max_length > hard_max:
        lengths *= float(hard_max) / max_length
    out_shape = (int(np.asscalar(lengths[0])),
                 int(np.asscalar(lengths[1])),
                 img_shape[2])
    return out_shape


def main(gpu=-1, epoch=100, batch_size=1, lr=5e-4, out='result'):
    train_data = VOCDetectionDataset(mode='train', use_cache=True, year='2007')
    test_data = VOCDetectionDataset(mode='val', use_cache=True, year='2007')

    def transform(in_data):
        img, bbox = in_data
        img -= np.array([103.939, 116.779, 123.68])[:, None, None]

        # Resize bounding box to a shape
        # with the smaller edge at least at length 600
        input_shape = img.shape[1:]
        output_shape = _shape_soft_min_hard_max(input_shape, 600, 1200)
        img = transforms.resize(img, output_shape)
        bbox = transforms.resize_bbox(bbox, input_shape, output_shape)

        # horizontally flip
        img, flips = transforms.random_flip(
            img, horizontal_flip=True, return_flip=True)
        h_flip = flips['h']
        bbox = transforms.flip_bbox(bbox, output_shape, h_flip)
        return img, bbox

    transforms.extend(train_data, transform)
    transforms.extend(test_data, transform)

    model = FasterRCNN(gpu=gpu)
    if gpu != -1:
        model.to_gpu(gpu)
        chainer.cuda.get_device(gpu).use()
    # optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    optimizer = chainer.optimizers.Adam(
        alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    train_iter = chainer.iterators.SerialIterator(test_data, batch_size=1)
    updater = ParallelUpdater(train_iter, optimizer, devices={'main': gpu})

    # updater = chainer.training.updater.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    log_interval = 20, 'iteration'
    val_interval = 3000, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'main/time',
         'main/rpn_loss_cls',
         'main/rpn_loss_bbox',
         'main/loss_cls',
         'main/loss_bbox']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # visualize training
    trainer.extend(
        extensions.PlotReport(
            ['main/rpn_loss_cls'],
            file_name='rpn_loss_cls.png'
        ),
        trigger=log_interval
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/rpn_loss_bbox'],
            file_name='rpn_loss_bbox.png'
        ),
        trigger=log_interval
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/loss_cls'],
            file_name='loss_cls.png'
        ),
        trigger=log_interval
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/loss_bbox'],
            file_name='loss_bbox.png'
        ),
        trigger=log_interval
    )
    trainer.extend(
        DetectionVisReport(
            range(10),  # visualize outputs for the first 10 data of test_data
            train_data,
            model,
            filename_base='detection_train',
            predict_func=model.predict_bboxes
        ),
        trigger=val_interval, invoke_before_training=True)
    trainer.extend(
        DetectionVisReport(
            range(10),  # visualize outputs for the first 10 data of test_data
            test_data,
            model,
            forward_func=model.predict_bboxes
        ),
        trigger=val_interval, invoke_before_training=True)

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    fire.Fire(main)
