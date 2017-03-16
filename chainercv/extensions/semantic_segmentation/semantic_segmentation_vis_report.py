import collections
import numpy as np
import os.path as osp
import six
import warnings

import chainer
from chainer.utils import type_check

from chainercv.transforms import chw_to_pil_image
from chainercv.utils import check_type
from chainercv.utils import forward

try:
    from matplotlib import pyplot as plot

    _available = True

except ImportError:
    _available = False


def _check_available():
    if not _available:
        warnings.warn('matplotlib is not installed on your environment, '
                      'so nothing will be plotted at this time. '
                      'Please install matplotlib to plot figures.\n\n'
                      '  $ pip install matplotlib\n')


class SemanticSegmentationVisReport(chainer.training.extension.Extension):

    """An extension that visualizes output for semantic segmentation task.

    This extension visualizes the predicted bounding boxes together with the
    ground truth bounding boxes.
    The examples used in visualization are selected by ids included in
    :obj:`indices`.

    This extension wraps three steps of operations needed to visualize output
    of the model. :obj:`dataset`,
    :obj:`predict_func` and :obj:`vis_transform` are used at each step.

    1. Getting an example.
        :meth:`dataset.__getitem__` returns tuple of arrays that are used as
        the arguments for :obj:`predict_func`.

        .. code:: python

            inputs = dataset[i]
            img, label = inputs

        :obj:`i` corresponds to an id included in :obj:`indices`.

    2. Predicting the output.
        Given the inputs, :meth:`predict_func` returns a prediction.
        :meth:`predict_func` should accept the first element of the tuple
        returned by the dataset, and returns a prediction.
        A batch axis is added to the input image when fed into the function
        as an argument. The prediction returned by the function can be with
        or without batch axis. The code below illustrates this step.

        .. code:: python

            img = inputs[0]  # first element of the tuple
            pred_label = predict_func(img[None])

    3. Converting input arrays for visualization.
        Given the inputs from :meth:`dataset.__getitem__`, a method
        :meth:`vis_transform` should convert them into visualizable forms.
        The values returned by :meth:`vis_transform` should be

        .. code:: python

            img, label = vis_transform(inputs)

        :obj:`img` should be an image which is in HWC format, RGB and
        :obj:`dtype==numpy.uint8`.

    The process can be illustrated in the following code.

    .. code:: python

        img, label = dataset[i]
        pred_label = predict_func(img[None])  # add batch axis to the image
        pred_label = pred_label[0]  # (B, 1, H, W) -> (1, H, W)
        vis_img, vis_label = vis_transform(inputs)  # (H, W, C) and (H, W, 1)

        # Visualization code
        # Uses (vis_img, vis_label) as the ground truth output
        # Uses (vis_img, pred_label) as the predicted output

    .. note::
        All images and labels that are obtained from the dataset should be
        in CHW format. This means that :obj:`img` and :obj:`label` are of
        shape :math:`(3, H, W)` and :math:`(1, H, W)`.

        The output of the model should be in BCHW or CHW format. More
        concretely, :obj:`pred_label` should be of shape
        :math:`(1, 1, H, W)` or :math:`(1, H, W)`.

        The output of :obj:`vis_transform` should be in HWC format.
        This means that :obj:`vis_img` and :obj:`vis_label` should be in
        shape :math:`(H, W, 3)` and :math:`(H, W, 1)`.

    .. note::
        All datasets prepared in :mod:`chainercv.datasets` should work
        out of the box with the default value of :obj:`vis_transform`.

        However, if the dataset has been extended by transformers,
        :obj:`vis_transform` needs to offset some transformations
        that are applied in order to achive a visual quality.
        For example, when the mean value is subtracted from input images,
        the mean value needs to be added back inside of :obj:`vis_transform`.

    Args:
        indices (list of ints or int): List of indices for data to be
            visualized
        dataset: Dataset class that produces inputs to :obj:`target`.
        target: Link object used for visualization
        n_class (int): number of labels including background, but excluding
            unknowns.
        filename_base (int): basename for saved image
        predict_func (callable): Callable that is used to predict the
            class label of the image. This function takes an image stored
            at the first element of the tuple returned by the dataset with
            batch dimension added.
            As an output, this function returns a predicted class label,
            which is of shape :math:`(1, 1, H, W)`.  Note that :math:`H` and
            :math:`W` are height and width of the input image.
            Also, the first axis, which is a batch axis,
            can be removed. Please see description on the step 2 of internal
            mechanics found above for more detail.
            If :obj:`predict_func = None`, then :meth:`model.__call__`
            method will be called.
        vis_transform (callable): A callable that is used to convert tuple of
            arrays returned by :obj:`dataset.__getitem__`. This function
            should return tuple of arrays which can be used for visualization.
            More detail can be found at the description on the step 3 of
            internal mechanics found above.

    """
    invoke_before_training = False

    def __init__(self, indices, dataset, target, n_class,
                 filename_base='semantic_seg', predict_func=None,
                 vis_transform=chw_to_pil_image):
        _check_available()

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
            predict_type.dtype.kind == 'i',
            predict_type.ndim == 4,
            predict_type.shape[0] == 1,
            predict_type.shape[1] == 1,
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

    @staticmethod
    def available():
        _check_available()
        return _available

    def __call__(self, trainer):
        if not _available:
            return

        for idx in self.indices:
            formated_filename_base = osp.join(trainer.out, self.filename_base)
            out_file = (formated_filename_base +
                        '_idx={}'.format(idx) +
                        '_iter={}'.format(trainer.updater.iteration) + '.jpg')

            inputs = self.dataset[idx]
            gt = inputs[1]
            self._check_type_dataset(inputs)

            pred_label = forward(self.target, inputs[0],
                                 forward_func=self.predict_func,
                                 expand_dim=True)
            if pred_label.ndim == 3:
                pred_label = pred_label[None]
            self._check_type_model(pred_label)
            pred_label = pred_label[0][0]  # (1, 1, H, W) -> (H, W)

            vis_transformed = self.vis_transform(inputs)
            self._check_type_vis_transformed(vis_transformed)
            vis_img = vis_transformed[0]

            # mask
            pred_label[gt[0] == -1] = -1

            # prepare label
            pred_label = _process_label(pred_label, self.n_class)
            gt_label = _process_label(gt[0], self.n_class)

            plot.subplot(2, 2, 1)
            plot.imshow(vis_img)
            plot.axis('off')
            plot.subplot(2, 2, 3)
            plot.imshow(pred_label, vmin=-1, vmax=21)
            plot.axis('off')
            plot.subplot(2, 2, 4)
            plot.imshow(gt_label, vmin=-1, vmax=21)
            plot.axis('off')
            plot.savefig(out_file)
            plot.close()


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(N=256):
    cmap = np.zeros((N, 3))
    for i in six.moves.range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def _process_label(label, n_class, bg_label=-1):
    colors = labelcolormap(n_class)
    label_viz = colors[label]
    # label 0 color: (0, 0, 0, 0) -> (0, 0, 0, 255)
    label_viz[label == 0] = 0
    # background label will be colored as (122, 122, 122)
    label_viz[label == bg_label] = np.array([122, 122, 122])
    return label_viz


if __name__ == '__main__':
    from chainercv.datasets import VOCSemanticSegmentationDataset
    from chainercv.utils.test_utils import ConstantReturnModel
    import mock
    import tempfile

    dataset = VOCSemanticSegmentationDataset()
    _, label = dataset[0]

    model = ConstantReturnModel(label[None])

    trainer = mock.MagicMock()
    out_dir = tempfile.mkdtemp()
    print('outdir ', out_dir)
    trainer.out = out_dir
    trainer.updater.iteration = 0
    extension = SemanticSegmentationVisReport([0], dataset, model, 21)
    extension(trainer)
