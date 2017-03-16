import collections
import os.path as osp
import warnings

import chainer
from chainer.utils import type_check

from chainercv.tasks import vis_bbox
from chainercv.transforms import chw_to_pil_image
from chainercv.utils.extension_utils import check_type
from chainercv.utils.extension_utils import forward

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


def _detection_vis_transform(xs):
    return chw_to_pil_image(xs[0]), xs[1], xs[2]


class DetectionVisReport(chainer.training.extension.Extension):

    """An extension that visualizes output of a detection model.

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
            img, bbox, label = inputs

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
            # pred_bbox can be either (1, R, 4) or (R, 4)
            # pred_label can be either (1, R) or (R,)
            pred_bbox, pred_label = predict_func(img[None])

    3. Converting input arrays for visualization.
        Given the inputs from :meth:`dataset.__getitem__`, a method
        :meth:`vis_transform` should convert them into visualizable forms.
        The values returned by :meth:`vis_transform` should be

        .. code:: python

            img, bbox, label = vis_transform(inputs)

        :obj:`img` should be an image which is in HWC format, RGB and
        :obj:`dtype==numpy.uint8`.

    The process can be illustrated in the following code.

    .. code:: python

        img, bbox, label = dataset[i]
        pred_bbox, pred_label = predict_func((img[None]))
        pred_bbox = pred_bbox[0]  # (B, R, 4) -> (R, 4)
        pred_label = pred_label[0]  # (B, R)  -> (R,)
        vis_img, vis_bbox, vis_label = vis_transform(inputs)
        # Visualization code
        # Uses (vis_img, vis_bbox, vis_label) as the ground truth output
        # Uses (vis_img, pred_bbox, pred_label) as the predicted output

    .. note::
        The bounding boxes are expected to be packed into a two dimensional
        tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
        bounding boxes in the image. The second axis represents attributes of
        the bounding box. They are
        :obj:`(x_min, y_min, x_max, y_max)`,
        where the four attributes are coordinates of the bottom left and the
        top right vertices.

        The labels are packed into a one dimensional tensor of shape
        :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
        These are integers that correspond to object ID of the dataset.

    .. note::
        All datasets prepared in :mod:`chainercv.datasets` should work
        out of the box with the default value of :obj:`vis_transform`,
        which is :obj:`chainercv.transforms.chw_to_pil_image_tuple`.
        However, if the dataset has been extended by transforms,
        :obj:`vis_transform` needs to offset some transformations
        that are applied for correct visualization.
        For example, when the mean value is subtracted from input images,
        the mean value needs to be added back in :obj:`vis_transform`.

    Args:
        indices (list of ints or int): List of indices for data to be
            visualized
        target: Link object used for visualization
        dataset: Dataset class that produces inputs to :obj:`target`.
        filename_base (int): basename for saved image
        predict_func (callable): Callable that is used to predict the
            bounding boxes of the image. This function takes an image stored
            at the first element of the tuple returned by the dataset with
            batch dimension added.
            As an output, this function returns bounding boxes and labels,
            which have shape :math:`(1, R, 4)` and :math:`(1, R)`
            respectively. :math:`R` is the number of bounding
            boxes. Also, the first axis, which is a batch axis, can be removed.
            Please see description on the step 2 of internal mechanics found
            above for more detail.
            If :obj:`predict_func = None`, then :meth:`model.__call__`
            method will be called.
        vis_transform (callable): A callable that is used to convert tuple of
            arrays returned by :obj:`dataset.__getitem__`. This function
            should return tuple of arrays which can be used for visualization.
            More detail can be found at the description on the step 3 of
            internal mechanics found above.

    """

    invoke_before_training = False

    def __init__(self, indices, dataset, target,
                 filename_base='detection', predict_func=None,
                 vis_transform=_detection_vis_transform):
        _check_available()

        if not isinstance(indices, collections.Iterable):
            indices = list(indices)
        self.dataset = dataset
        self.target = target
        self.indices = indices
        self.filename_base = filename_base
        self.predict_func = predict_func
        self.vis_transform = vis_transform

    @check_type
    def _check_type_dataset(self, in_types):
        img_type = in_types[0]
        bbox_type = in_types[1]
        label_type = in_types[2]
        type_check.expect(
            img_type.shape[0] == 3,
            bbox_type.shape[1] == 4,
            img_type.ndim == 3,
            bbox_type.ndim == 2,
            label_type.ndim == 1,
            bbox_type.shape[0] == label_type.shape[0],
        )

    @check_type
    def _check_type_model(self, in_types):
        predict_bbox_type = in_types[0]
        predict_label_type = in_types[1]
        type_check.expect(
            predict_bbox_type.ndim == 3,
            predict_label_type.ndim == 2,
            predict_bbox_type.shape[0] == 1,
            predict_bbox_type.shape[2] == 4,
            predict_label_type.shape[0] == 1,
            predict_bbox_type.shape[1] == predict_label_type.shape[1],
        )

    @check_type
    def _check_type_vis_transformed(self, in_types):
        img_type = in_types[0]
        bbox_type = in_types[1]
        label_type = in_types[2]
        type_check.expect(
            img_type.dtype.kind == 'u',
            img_type.ndim == 3,
            img_type.shape[2] == 3,
            bbox_type.ndim == 2,
            bbox_type.shape[1] == 4,
            label_type.ndim == 1,
            bbox_type.shape[0] == label_type.shape[0],
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
            self._check_type_dataset(inputs)

            pred_bbox, pred_label = forward(self.target, inputs[0],
                                            forward_func=self.predict_func,
                                            expand_dim=True)
            if pred_bbox.ndim == 2:
                # force output to have batch axis
                pred_bbox = pred_bbox[None]
                pred_label = pred_label[None]
            self._check_type_model((pred_bbox, pred_label))
            pred_bbox = pred_bbox[0]  # (B, R, 5) -> (R, 5)

            vis_transformed = self.vis_transform(inputs)
            self._check_type_vis_transformed(vis_transformed)
            vis_img = vis_transformed[0]
            raw_bbox = vis_transformed[1]

            # start visualizing using matplotlib
            fig = plot.figure()

            ax_gt = fig.add_subplot(2, 1, 1)
            ax_gt.set_title('ground truth')
            label_names = getattr(self.dataset, 'labels', None)
            vis_bbox(
                vis_img, raw_bbox, label_names=label_names, ax=ax_gt)

            ax_pred = fig.add_subplot(2, 1, 2)
            ax_pred.set_title('prediction')

            vis_bbox(vis_img, pred_bbox, label_names=label_names, ax=ax_pred)

            plot.savefig(out_file)
            plot.close()


if __name__ == '__main__':
    from chainercv.datasets import VOCDetectionDataset
    from chainercv.utils.test_utils import ConstantReturnModel
    import mock
    import tempfile
    train_data = VOCDetectionDataset(mode='train', year='2007')
    _, bbox, label = train_data[3]

    model = ConstantReturnModel((bbox[None], label[None]))

    trainer = mock.MagicMock()
    out_dir = tempfile.mkdtemp()
    print('outdir ', out_dir)
    trainer.out = out_dir
    trainer.updater.iteration = 0
    extension = DetectionVisReport([3], train_data, model)
    extension(trainer)
