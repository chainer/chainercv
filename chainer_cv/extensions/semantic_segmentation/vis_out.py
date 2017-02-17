import collections
import numpy as np
import os
import os.path as osp
from skimage.color import label2rgb

import chainer
from chainer.training import extensions
from chainer.utils import type_check

from chainer_cv.extensions.utils import forward


class SemanticSegmentationVisOut(chainer.training.extension.Extension):
    """An extension that visualizes input and output of semantic segmentation

    This extension visualizes predicted label, ground truth label and input
    image.

    The model that is used for forwarding is obtained by the command below.
    model = trainer.updater.get_optimizer('main').target

    Args:
        indices (list of int): List of indices for data to be visualized
        n_class (int): number of classes
        filename_base (int): basename for saved image
        forward_func (callable): Callable that is used to forward data input.
            This callable takes all the arrays returned by the dataset as
            input. Also, this callable returns an prediction of labels. 
            If `forward_func = None`, then the model's `__call__` method will
            be called.

    """
    invoke_before_training = False

    def __init__(self, indices, n_class, filename_base='semantic_seg_train',
                 forward_func=None):
        if not isinstance(indices, collections.Iterable):
            indices = list(indices)
        self.indices = indices
        self.n_class = n_class
        self.filename_base = filename_base
        self.forward_func = forward_func
    
        out_folder = osp.split(filename_base)[0]
        if not osp.exists(out_folder):
            os.makedirs(out_folder)

    def _typecheck_dataset(self, args):
        assert(args[0].ndim == args[1].ndim == 3)
        assert(args[0].shape[0] == 3 and args[1].shape[0] == 1)
        assert(args[0].shape[1] == args[1].shape[1])
        assert(args[0].shape[2] == args[1].shape[2])
        assert(np.issubdtype(args[0].dtype, np.float))
        assert(np.issubdtype(args[1].dtype, np.int))

    def _typecheck_model(self, args):
        assert(args[0].ndim == 4)
        assert(args[0].shape[0] == 1)
        assert(args[0].shape[1] == self.n_class)

    def _typecheck_get_raw_img(self, args):
        assert(args[0].ndim == 3)
        assert(args[1].ndim == 2)
        assert(args[0].shape[2] == 3)

    def __call__(self, trainer):
        import matplotlib.pyplot as plt
        model = trainer.updater.get_optimizer('main').target
        dataset = trainer.updater.get_iterator('main').dataset
        for idx in self.indices:
            formated_filename_base = osp.join(trainer.out, self.filename_base)
            out_file = (formated_filename_base +
                        'idx=_{}'.format(idx) +
                        'iter={}'.format(trainer.updater.iteration) + '.jpg')

            inputs = dataset[idx]
            gt = inputs[1]
            self._typecheck_dataset(inputs)
            out = forward(model, inputs, forward_func=self.forward_func)
            self._typecheck_model(out)
            label = np.argmax(out[0][0], axis=0)

            if not hasattr(dataset, 'get_raw_img'):
                raise ValueError(
                    'the dataset class needs to have a method '
                    '``get_raw_img`` for a visualization extension')
            raw_inputs = dataset.get_raw_img(idx)
            self._typecheck_get_raw_img(raw_inputs)
            vis_img = raw_inputs[0]

            # mask
            label[gt[0] == -1] = -1
            
            # prepare label
            x_slices, y_slices = _get_pad_slices(gt[0], unknown_val=-1)
            label = _process_label(label, self.n_class)
            gt_label = _process_label(gt[0], self.n_class)
            label = label[y_slices, x_slices]
            gt_label = gt_label[y_slices, x_slices]

            plt.close()
            plt.subplot(2, 2, 1)
            plt.imshow(vis_img)
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(label, vmin=-1, vmax=21)
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(gt_label, vmin=-1, vmax=21)
            plt.axis('off')
            plt.savefig(out_file)


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(N=256):
    cmap = np.zeros((N, 3))
    for i in xrange(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in xrange(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def _process_label(label, n_class, bg_label=0):
    colors = labelcolormap(n_class)
    label_viz = label2rgb(label, image=None, colors=colors[1:], bg_label=bg_label)
    # label 0 color: (0, 0, 0, 0) -> (0, 0, 0, 255)
    label_viz[label == 0] = 0
    return label_viz


def _get_pad_slices(label, unknown_val=-1):
    where_val = np.where(label != -1)
    y = where_val[0]
    x = where_val[1]
    y_slices = slice(np.min(y), np.max(y) + 1, None)
    x_slices = slice(np.min(x), np.max(x) + 1, None)
    return x_slices, y_slices
