import matplotlib.pyplot as plt

import collections
import numpy as np
import os.path as osp

import chainer
from chainer import training
from chainer.training import extensions
import chainer.optimizers as O

from chainer_cv.extensions.semantic_segmentation.draw_label import draw_label


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
    from skimage.color import label2rgb
    colors = labelcolormap(n_class)
    label_viz = label2rgb(label, image=None, colors=colors[1:], bg_label=bg_label)
    # label 0 color: (0, 0, 0, 0) -> (0, 0, 0, 255)
    label_viz[label == 0] = 0
    return label_viz


def _forward(model, img):
    xp = model.xp
    if xp == np:
        raise ValueError
    else:
        img_var = chainer.Variable(img[None, :, :, :])
        img_var.to_gpu()
        feat_var = model(img_var)[0]
        feat = chainer.cuda.to_cpu(feat_var.data)
    return feat


def get_pad_slices(label, unknown_val=-1):
    where_val = np.where(label != -1)
    y = where_val[0]
    x = where_val[1]
    y_slices = slice(np.min(y), np.max(y) + 1, None)
    x_slices = slice(np.min(x), np.max(x) + 1, None)
    return x_slices, y_slices


def extension_make_img(indices, n_class, filename_base):
    """
    Args:
        indices (list of data ids)
    """
    @chainer.training.make_extension()
    def make_img(trainer):
        if not isinstance(indices, collections.Iterable):
            local_indices = list(indices)
        else:
            local_indices = indices

        model = trainer.updater.get_optimizer('main').target
        dataset = trainer.updater.get_iterator('main').dataset
        for idx in local_indices:
            img, gt = dataset[idx]

            formated_filename_base = filename_base.format(trainer)
            out_file = (formated_filename_base +
                        'idx=_{}'.format(idx) +
                        'iter={}'.format(trainer.updater.iteration) + '.jpg')
            out_folder = osp.split(out_file)[0]
            if not osp.exists(out_folder):
                os.makedirs(out_folder)

            out = _forward(model, img)
            label = np.argmax(out, axis=0)
            # mask
            label[gt[0] == -1] = -1

            vis_img = dataset.get_vis_img(idx)
            
            # prepare label
            x_slices, y_slices = get_pad_slices(gt[0], unknown_val=-1)
            label = _process_label(label, n_class)
            gt_label = _process_label(gt[0], n_class)
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
    return make_img
