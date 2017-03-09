import collections
import numpy as np
import os.path as osp
import warnings

import chainer
from chainer.utils import type_check

from chainercv.transforms import chw_to_pil_image_tuple
from chainercv.utils import check_type
from chainercv.utils import forward

from matplotlib import pyplot as plt

try:
    from skimage.color import label2rgb
    _available = True

except ImportError:
    _available = False


def chw_to_pil_image_tuple_img_label(xs):
    return chw_to_pil_image_tuple(xs, indices=[0, 1])


class SemanticSegmentationVisReport(chainer.training.extension.Extension):

    """An extension that visualizes output for semantic segmentation task.

    This extension visualizes the predicted bounding boxes together with the
    ground truth bounding boxes.
    The examples used in visualization are selected by ids included in
    :obj:`indices`.

    This extension wraps three steps of operations needed to visualize output
    of the model. :obj:`dataset`,
    :obj:`predict_func` and :obj:`vis_transformer` are used at each step.

    1. Getting an example.
        :meth:`dataset.__getitem__` returns tuple of arrays that are used as
        the arguments for :obj:`predict_func`.

        .. code:: python

            inputs = dataset[i]
            img, label = inputs

        :obj:`i` corresponds to an id included in :obj:`indices`.

    2. Predicting the output.
        Given the inputs, :meth:`predict_func` returns the tuple of arrays as
        outputs. :meth:`predict_func` should accept inputs with a batch axis
        and returns outputs with a batch axis. The function should be used
        like below.

        .. code:: python

            img, label = inputs
            pred_label, = predict_func((img[None], label[None]))

    3. Converting input arrays for visualization.
        Given the inputs from :meth:`dataset.__getitem__`, a method
        :meth:`vis_transformer` should convert them into visualizable forms.
        The values returned by :meth:`vis_transformer` should be

        .. code:: python

            img, label = vis_transformer(inputs)

        :obj:`img` should be an image which is in HWC format, RGB and
        :obj:`dtype==numpy.uint8`.

    The process can be illustrated in the following code.

    .. code:: python

        img, label = dataset[i]
        pred_label, = predict_func((img[None], label[None])  # add batch axis
        pred_label = pred_label[0]  # remove batch axis
        vis_img, vis_label = vis_transformer(inputs)

        # Visualization code
        # Uses (vis_img, vis_label) as the ground truth output
        # Uses (vis_img, pred_label) as the predicted output

    .. note::
        All images and labels that are obtained from the dataset should be
        in CHW format. This means that :obj:`img` and :obj:`label` are of
        shape :math:`(3, H, W)` and :math:`(1, H, W)`.

        The output of the model should be in BCHW format. More concretely,
        :obj:`pred_label` should be of shape :math:`(1, L, H, W)`. Note that
        :math:`L` is number of categories including the background.

        The output of :obj:`vis_transformer` should be in HWC format.
        This means that :obj:`vis_img` and :obj:`vis_label` should be in
        shape :math:`(H, W, 3)` and :math:`(H, W, 1)`.

    .. note::
        All datasets prepared in :mod:`chainercv.datasets` should work
        out of the box with the default value of :obj:`vis_transformer`.

        However, if the dataset has been extended by transformers,
        :obj:`vis_transformer` needs to offset some transformations
        that are applied in order to achive a visual quality.
        For example, when the mean value is subtracted from input images,
        the mean value needs to be added back inside of :obj:`vis_transformer`.

    Args:
        indices (list of ints or int): List of indices for data to be
            visualized
        target: Link object used for visualization
        dataset: Dataset class that produces inputs to :obj:`target`.
        filename_base (int): basename for saved image
        predict_func (callable): Callable that is used to forward data input.
            This callable takes all the arrays returned by the dataset as
            input. Also, this callable returns an predicted bounding boxes.
            If :obj:`predict_func = None`, then :meth:`model.__call__`
            method will be called.
        vis_transformer (callable): A callable that is used to convert tuple of
            arrays returned by :obj:`dataset.__getitem__`. This function
            should return tuple of arrays which can be used for visualization.

    """
    invoke_before_training = False

    def __init__(self, indices, dataset, target, n_class,
                 filename_base='semantic_seg', predict_func=None,
                 vis_transform=chw_to_pil_image_tuple_img_label):
        if not isinstance(indices, collections.Iterable):
            indices = list(indices)
        self.dataset = dataset
        self.target = target
        self.indices = indices
        self.n_class = n_class
        self.filename_base = filename_base
        self.predict_func = predict_func
        self.vis_transform = vis_transform

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
    def _check_type_vis_transformed(self, in_types):
        img_type = in_types[0]
        label_type = in_types[1]
        type_check.expect(
            img_type.ndim == 3,
            label_type.ndim == 3,
            img_type.shape[2] == 3,
            label_type.shape[2] == 1,
        )

    def __call__(self, trainer):
        if not _available:
            warnings.warn('scikit-image is not installed on your environment, '
                          'so a function embedding_tensorboard can not be '
                          ' used. Please install scikit-image.\n\n'
                          '  $ pip install scikit-image\n')
            return

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

            vis_transformed = self.vis_transform(inputs)
            self._check_type_vis_transformed(vis_transformed)
            vis_img = vis_transformed[0]

            # mask
            label[gt[0] == -1] = -1

            # prepare label
            label = _process_label(label, self.n_class)
            gt_label = _process_label(gt[0], self.n_class)

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
            plt.close()


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
