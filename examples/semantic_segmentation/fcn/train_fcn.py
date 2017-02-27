import argparse
import os.path as osp

import chainer
from chainer import training
from chainer.training import extensions

from chainer_cv.datasets import VOCSemanticSegmentationDataset
from chainer_cv.extensions import SemanticSegmentationVisReport
from chainer_cv.training.test_mode_evaluator import TestModeEvaluator
from chainer_cv.wrappers import PadWrapper
from chainer_cv.wrappers import SubtractWrapper

from fcn32s import FCN32s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('-it', '--iterations', type=int, default=100000)
    parser.add_argument('-ba', '--batch-size', type=int, default=1)
    parser.add_argument('-l', '--lr', type=float, default=1e-10)
    parser.add_argument('-o', '--out', type=str, default='result')

    args = parser.parse_args()
    gpu = args.gpu
    batch_size = args.batch_size
    iterations = args.iterations
    resume = args.resume
    lr = args.lr
    out = args.out

    # prepare datasets
    wrappers = [lambda d: SubtractWrapper(d),
                lambda d: PadWrapper(
                    d, max_size=(512, 512), preprocess_idx=[0, 1],
                    bg_values={0: 0, 1: -1})]
    train_data = VOCSemanticSegmentationDataset(mode='train')
    test_data = VOCSemanticSegmentationDataset(mode='val')
    for wrapper in wrappers:
        train_data = wrapper(train_data)
        test_data = wrapper(test_data)

    # set up FCN32s
    n_class = 21
    model = FCN32s(n_class=n_class)
    if gpu != -1:
        model.to_gpu(gpu)
        chainer.cuda.get_device(gpu).use()

    # prepare an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.99)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # prepare iterators
    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=batch_size)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (iterations, 'iteration'), out=out)

    val_interval = 3000, 'iteration'
    log_interval = 100, 'iteration'

    trainer.extend(
        TestModeEvaluator(test_iter, model, device=gpu), trigger=val_interval)

    # reporter related
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'main/time',
         'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy',
         'main/accuracy_cls', 'validation/main/accuracy_cls',
         'main/iu', 'validation/main/iu',
         'main/fwavacc', 'validation/main/fwavacc']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # visualize training
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
        SemanticSegmentationVisReport(
            range(10),  # visualize outputs for the first 10 data of test_data
            test_data,
            model,
            n_class=n_class,
            predict_func=model.extract  # use FCN32s.extract to get a score map
        ),
        trigger=val_interval, invoke_before_training=True)

    trainer.extend(extensions.dump_graph('main/loss'))

    if resume:
        chainer.serializers.load_npz(osp.expanduser(resume), trainer)

    trainer.run()
