import collections
import os.path as osp

import chainer
from chainer.utils import type_check

from chainercv.tasks.detection import vis_img_bbox
from chainercv.transforms import chw_to_pil_image_tuple
from chainercv.utils.extension_utils import check_type
from chainercv.utils.extension_utils import forward

from matplotlib import pyplot as plt


class DetectionVisReport(chainer.training.extension.Extension):

    """An extension that visualizes output of a detection model.

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
            img, bbox = inputs

        :obj:`i` corresponds to an id included in :obj:`indices`.

    2. Predicting the output.
        Given the inputs, :meth:`predict_func` returns the tuple of arrays as
        outputs. :meth:`predict_func` should accept inputs with a batch axis
        and returns outputs with a batch axis. The function should be used
        like below.

        .. code:: python

            img, bbox = inputs
            pred_bbox, = predict_func((img[None], bbox[None]))

    3. Converting input arrays for visualization.
        Given the inputs from :meth:`dataset.__getitem__`, a method
        :meth:`vis_transformer` should convert them into visualizable forms.
        The values returned by :meth:`vis_transformer` should be

        .. code:: python

            img, bbox = vis_transformer(inputs)

        :obj:`img` should be an image which is in HWC format, RGB and
        :obj:`dtype==numpy.uint8`.

    The process can be illustrated in the following code.

    .. code:: python

        img, bbox = dataset[i]
        pred_bbox, = predict_func((img[None], bbox[None])  # add batch axis
        pred_bbox = pred_bbox[0]  # remove batch axis
        vis_img, vis_bbox = vis_transformer(inputs)

        # Visualization code
        # Uses (vis_img, vis_bbox) as the ground truth output
        # Uses (vis_img, pred_bbox) as the predicted output

    .. note::
        The bounding box is expected to be a two dimensional tensor of shape
        :math:`(R, 5)`, where :math:`R` is the number of bounding boxes in
        the image. The second axis represents attributes of the bounding box.
        They are :obj:`(x_min, y_min, x_max, y_max, label_id)`, where first
        four attributes are coordinates of the bottom left and the top right
        vertices. The last attribute is the label id, which points to the
        category of the object in the bounding box.

    .. note::
        All datasets prepared in :mod:`chainercv.datasets` should work
        out of the box with the default value of :obj:`vis_transformer`,
        which is :obj:`chainercv.transforms.chw_to_pil_image_tuple`.

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

    def __init__(self, indices, dataset, target,
                 filename_base='detection', predict_func=None,
                 vis_transform=chw_to_pil_image_tuple):
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
        bboxes_type = in_types[1]
        type_check.expect(
            img_type.shape[0] == 3,
            bboxes_type.shape[1] == 5,
            img_type.ndim == 3,
            bboxes_type.ndim == 2
        )

    @check_type
    def _check_type_model(self, in_types):
        predict_bboxes_type = in_types[0]
        type_check.expect(
            predict_bboxes_type.ndim == 3,
            predict_bboxes_type.shape[0] == 1,
            predict_bboxes_type.shape[2] == 5,
        )

    @check_type
    def _check_type_vis_transformed(self, in_types):
        img_type = in_types[0]
        bboxes_type = in_types[1]
        type_check.expect(
            img_type.dtype.kind == 'u',
            img_type.ndim == 3,
            img_type.shape[2] == 3,
            bboxes_type.ndim == 2,
            bboxes_type.shape[1] == 5
        )

    def __call__(self, trainer):
        for idx in self.indices:
            formated_filename_base = osp.join(trainer.out, self.filename_base)
            out_file = (formated_filename_base +
                        '_idx={}'.format(idx) +
                        '_iter={}'.format(trainer.updater.iteration) + '.jpg')

            inputs = self.dataset[idx]
            self._check_type_dataset(inputs)

            if hasattr(self.target, 'train'):
                original = self.target.train
                self.target.train = False
            out = forward(self.target, inputs,
                          forward_func=self.predict_func, expand_dim=True)
            if hasattr(self.target, 'train'):
                self.target.train = original
            self._check_type_model(out)
            bboxes = out[0][0]  # (R, 5)

            vis_transformed = self.vis_transform(inputs)
            self._check_type_vis_transformed(vis_transformed)
            vis_img = vis_transformed[0]
            raw_bboxes = vis_transformed[1]

            # start visualizing using matplotlib
            fig = plt.figure()

            ax_gt = fig.add_subplot(2, 1, 1)
            ax_gt.set_title('ground truth')
            label_names = getattr(self.dataset, 'labels', None)
            vis_img_bbox(
                vis_img, raw_bboxes, label_names=label_names, ax=ax_gt)

            ax_pred = fig.add_subplot(2, 1, 2)
            ax_pred.set_title('prediction')
            vis_img_bbox(vis_img, bboxes, label_names=label_names, ax=ax_pred)

            plt.savefig(out_file)
            plt.close()


if __name__ == '__main__':
    from chainercv.datasets import VOCDetectionDataset
    from chainercv.testing import ConstantReturnModel
    import mock
    import tempfile
    train_data = VOCDetectionDataset(mode='train', use_cache=True, year='2007')
    _, bbox = train_data.get_example(3)

    model = ConstantReturnModel(bbox[None])

    trainer = mock.MagicMock()
    out_dir = tempfile.mkdtemp()
    print('outdir ', out_dir)
    trainer.out = out_dir
    trainer.updater.iteration = 0
    extension = DetectionVisReport([3], train_data, model)
    extension(trainer)
