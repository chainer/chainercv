import fire
import numpy as np
import os.path as osp

import chainer
from chainer import training
from chainer.training import extensions

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.extensions import SemanticSegmentationVisReport
from chainercv import transforms

from fcn32s import FCN32s


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main(gpu=-1, batch_size=1, iterations=100000,
         lr=1e-10, out='result', resume=''):
    # prepare datasets
    def transform(in_data):
        img, label = in_data
        vgg_subtract_bgr = np.array(
            [103.939, 116.779, 123.68], np.float32)[:, None, None]
        img -= vgg_subtract_bgr
        img = transforms.pad(img, max_size=(512, 512), bg_value=0)
        label = transforms.pad(label, max_size=(512, 512), bg_value=-1)
        return img, label

    train_data = VOCSemanticSegmentationDataset(mode='train')
    test_data = VOCSemanticSegmentationDataset(mode='val')
    transforms.extend(train_data, transform)
    transforms.extend(test_data, transform)

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

    def vis_transform(in_data):
        vgg_subtract_bgr = np.array(
            [103.939, 116.779, 123.68], np.float32)[:, None, None]
        img, label = in_data
        img += vgg_subtract_bgr
        img, label = transforms.chw_to_pil_image_tuple(
            (img, label), indices=[0, 1])
        return img, label

    trainer.extend(
        SemanticSegmentationVisReport(
            range(10),  # visualize outputs for the first 10 data of test_data
            test_data,
            model,
            n_class=n_class,
            predict_func=model.extract,  # use FCN32s.extract to get a scoremap
            vis_transform=vis_transform
        ),
        trigger=val_interval, invoke_before_training=True)

    trainer.extend(extensions.dump_graph('main/loss'))

    if resume:
        chainer.serializers.load_npz(osp.expanduser(resume), trainer)

    trainer.run()


if __name__ == '__main__':
    fire.Fire(main)
