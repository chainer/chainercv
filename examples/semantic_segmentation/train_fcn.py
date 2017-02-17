import argparse
import os.path as osp

import chainer
from chainer import training
from chainer.training import extensions

from chainer_cv.training.test_mode_evaluator import TestModeEvaluator
from chainer_cv.datasets import PascalVOCDataset
from chainer_cv.wrappers import PadWrapper
from chainer_cv.wrappers import SubtractWrapper
from chainer_cv.extensions.semantic_segmentation.vis_out import SemanticSegmentationVisOut

from fcn32s import FCN32s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('-ne', '--epochs', type=int, default=20000)
    parser.add_argument('-ba', '--batch-size', type=int, default=3)
    parser.add_argument('-l', '--lr', type=float, default=1e-10)
    parser.add_argument('-o', '--outdir', type=str, default='result')

    args = parser.parse_args()
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    resume = args.resume
    lr = args.lr
    outdir = args.outdir

    train_data = PadWrapper(SubtractWrapper(PascalVOCDataset(mode='train')))
    test_data = PadWrapper(SubtractWrapper(PascalVOCDataset(mode='val')))

    n_class = 21
    model = FCN32s(n_class=n_class)

    if gpu != -1:
        model.to_gpu(gpu)
        chainer.cuda.get_device(gpu).use()

    # optimizer = O.Adam(alpha=1e-9)
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.99)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.GradientClipping(10.))

    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=batch_size, n_processes=2,
        shared_mem=10000000
    )
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=outdir)

    val_interval = 3000, 'iteration'
    log_interval = 100, 'iteration'

    trainer.extend(
        TestModeEvaluator(test_iter, model, device=gpu), trigger=val_interval)

    # reporter related
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/time',
         'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy',
         'main/accuracy_cls', 'validation/main/accuracy_cls',
         'main/iu', 'validation/main/iu',
         'main/fwavacc', 'validation/main/fwavacc']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # training visualization
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            trigger=log_interval, file_name='loss.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            trigger=log_interval, file_name='accuracy.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy_cls', 'validation/main/accuracy_cls'],
            trigger=log_interval, file_name='accuracy_cls.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/iu', 'validation/main/iu'],
            trigger=log_interval, file_name='iu.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/fwavacc', 'validation/main/fwavacc'],
            trigger=log_interval, file_name='fwavacc.png')
    )
    trainer.extend(
        SemanticSegmentationVisOut(
            range(10),
            n_class=n_class,
            forward_func=model.extract
        ),
        trigger=val_interval, invoke_before_training=True)

    trainer.extend(extensions.dump_graph('main/loss'))

    if resume:
        chainer.serializers.load_npz(osp.expanduser(resume), trainer)

    trainer.run()
