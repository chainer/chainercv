import collections
import numpy as np
import os.path as osp
from skimage.color import label2rgb

import chainer
from chainer.utils import type_check

from chainer_cv.extensions.utils import check_type
from chainer_cv.extensions.utils import forward

from matplotlib import pyplot as plt


class SemanticSegmentationVisReport(chainer.training.extension.Extension):
    """An extension that visualizes input and output of semantic segmentation.

    This extension visualizes predicted label, ground truth label and input
    image.

    Args:
        indices (list of ints or int): List of indices for data to be
            visualized
        target: Link object used for visualization
        dataset: Dataset class that produces inputs to ``target``.
        n_class (int): number of classes
        filename_base (int): basename for saved image
        predict_func (callable): Callable that is used to forward data input.
            This callable takes all the arrays returned by the dataset as
            input. Also, this callable returns an prediction of labels.
            If `predict_func = None`, then the model's `__call__` method will
            be called.

    """
    invoke_before_training = False

    def __init__(self, indices, dataset, target, n_class,
                 filename_base='semantic_seg', predict_func=None):
        if not isinstance(indices, collections.Iterable):
            indices = list(indices)
        self.dataset = dataset
        self.target = target
        self.indices = indices
        self.n_class = n_class
        self.filename_base = filename_base
        self.predict_func = predict_func

    @check_type
    def _check_type_dataset(self, in_types):
        img_type = in_types[0]
        label_type = in_types[1]
        type_check.expect(
            img_type.dtype.kind == 'f',
            label_type.dtype.kind == 'i',
            img_type.shape[0] == 3,
            label_type.shape[0] == 1,
            img_type.shape[1] == label_type.shape[1],
            img_type.shape[2] == label_type.shape[2],
            img_type.ndim == 3,
            label_type.ndim == 3
        )

    @check_type
    def _check_type_model(self, in_types):
        predict_type = in_types[0]
        type_check.expect(
            predict_type.ndim == 4,
            predict_type.shape[0] == 1,
            predict_type.shape[1] == self.n_class
        )

    @check_type
    def _check_type_get_raw_data(self, in_types):
        img_type = in_types[0]
        label_type = in_types[1]
        type_check.expect(
            img_type.ndim == 3,
            label_type.ndim == 2,
            img_type.shape[2] == 3
        )

    def __call__(self, trainer):
        for idx in self.indices:
            formated_filename_base = osp.join(trainer.out, self.filename_base)
            out_file = (formated_filename_base +
                        '_idx={}'.format(idx) +
                        '_iter={}'.format(trainer.updater.iteration) + '.jpg')

            inputs = self.dataset[idx]
            gt = inputs[1]
            self._check_type_dataset(inputs)
            out = forward(self.target, inputs,
                          forward_func=self.predict_func, expand_dim=True)
            self._check_type_model(out)
            label = np.argmax(out[0][0], axis=0)

            if not hasattr(self.dataset, 'get_raw_data'):
                raise ValueError(
                    'the dataset class needs to have a method '
                    '``get_raw_data`` for a visualization extension')
            raw_inputs = self.dataset.get_raw_data(idx)
            self._check_type_get_raw_data(raw_inputs)
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
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def _process_label(label, n_class, bg_label=0):
    colors = labelcolormap(n_class)
    label_viz = label2rgb(
        label, image=None, colors=colors[1:], bg_label=bg_label)
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
