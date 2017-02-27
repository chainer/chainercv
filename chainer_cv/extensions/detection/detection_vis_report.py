import collections
import os.path as osp

import chainer
from chainer.utils import type_check

from chainer_cv.extensions.utils import check_type
from chainer_cv.extensions.utils import forward
from chainer_cv.visualizations import vis_img_bbox

from matplotlib import pyplot as plt


class DetectionVisReport(chainer.training.extension.Extension):
    """An extension that visualizes output of a detection model.

    This extension visualizes the predicted bounding boxes together with the
    ground truth bounding boxes.

    Args:
        indices (list of ints or int): List of indices for data to be
            visualized
        target: Link object used for visualization
        dataset: Dataset class that produces inputs to ``target``.
        filename_base (int): basename for saved image
        predict_func (callable): Callable that is used to forward data input.
            This callable takes all the arrays returned by the dataset as
            input. Also, this callable returns an predicted bounding boxes.
            If `predict_func = None`, then the model's `__call__` method will
            be called.

    """
    invoke_before_training = False

    def __init__(self, indices, dataset, target,
                 filename_base='detection', predict_func=None):
        if not isinstance(indices, collections.Iterable):
            indices = list(indices)
        self.dataset = dataset
        self.target = target
        self.indices = indices
        self.filename_base = filename_base
        self.predict_func = predict_func

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
    def _check_type_get_raw_data(self, in_types):
        img_type = in_types[0]
        bboxes_type = in_types[1]
        type_check.expect(
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
            input_img = inputs[0]

            if hasattr(self.target, 'train'):
                original = self.target.train
                self.target.train = False
            out = forward(self.target, inputs,
                          forward_func=self.predict_func, expand_dim=True)
            if hasattr(self.target, 'train'):
                self.target.train = original
            self._check_type_model(out)
            bboxes = out[0][0]  # (R, 5)

            if not hasattr(self.dataset, 'get_raw_data'):
                raise ValueError(
                    'the dataset class needs to have a method '
                    '``get_raw_data`` for a visualization extension')
            raw_inputs = self.dataset.get_raw_data(idx)
            self._check_type_get_raw_data(raw_inputs)
            vis_img = raw_inputs[0]
            raw_bboxes = raw_inputs[1]

            H, W = input_img.shape[1:]
            raw_H, raw_W = vis_img.shape[:2]
            scale = float(H) / raw_H
            assert abs(scale - (float(W) / raw_W)) < 0.02
            bboxes[:, :4] = bboxes[:, :4] / scale

            plt.close()
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


if __name__ == '__main__':
    from chainer_cv.datasets import VOCDetectionDataset
    from chainer_cv.testing import ConstantReturnModel
    import mock
    import tempfile
    train_data = VOCDetectionDataset(mode='train', use_cache=True, year='2007')
    _, bbox = train_data.get_example(3)

    model = ConstantReturnModel(bbox)

    trainer = mock.MagicMock()
    out_dir = tempfile.mkdtemp()
    print('outdir ', out_dir)
    trainer.out = out_dir
    trainer.updater.iteration = 0
    extension = DetectionVisReport([3], train_data, model)
    extension(trainer)
