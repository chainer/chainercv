import copy
import os
import warnings

import chainer

from chainercv.visualizations.vis_bbox import vis_bbox

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


class DetectionVisReport(chainer.training.extension.Extension):

    """An extension that visualizes output of a detection model.

    This extension visualizes the predicted bounding boxes together with the
    ground truth bounding boxes.

    The process can be illustrated in the following code.

    .. code:: python

        img, gt_bbox, gt_label = dataset[i]
        pred_bbox, pred_label, pred_score = target.predict(img)
        # Visualization code
        # Uses (img, gt_bbox, gt_label) as the ground truth output
        # Uses (img, pred_bbox, pred_label, pred_score) as the predicted output

    .. note::
        :obj:`gt_bbox` and :obj:`pred_bbox` are float arrays
        of shape :math:`(R, 4)`, where :math:`R` is the number of
        bounding boxes in the image. Each bounding box is organized
        by :obj:`(x_min, y_min, x_max, y_max)` in the second axis.

        :obj:`gt_label` and :obj:`pred_label` are intenger arrays
        of shape :math:`(R,)`. Each label indicates the class of
        the bounding box.

        :obj:`pred_score` is a float array of shape :math:`(R,)`.
        Each score indicates how confident the prediction is.

    Args:
        target (chainercv.links.DetectionLink): Link object used for detection.
        dataset : Dataset class that produces inputs to :obj:`target`.
        filename (str): Basename for saved image.
    """

    invoke_before_training = False

    def __init__(
            self, iterator, target, label_names=None,
            filename='detection_iter={iteration}_idx={index}.jpg'):
        _check_available()

        self.iterator = iterator
        self.target = target
        self.label_names = label_names
        self.filename = filename

    @staticmethod
    def available():
        _check_available()
        return _available

    def __call__(self, trainer):
        if not _available:
            return

        if hasattr(self.iterator, 'reset'):
            self.iterator.reset()
            it = self.iterator
        else:
            it = copy.copy(self.iterator)

        idx = 0
        while True:
            try:
                batch = next(it)
            except StopIteration:
                break

            imgs = [img for img, _, _ in batch]
            pred_bboxes, pred_labels, pred_scores = self.target.predict(imgs)

            for (img, gt_bbox, gt_label), pred_bbox, pred_label, pred_score \
                    in zip(batch, pred_bboxes, pred_labels, pred_scores):

                pred_bbox = chainer.cuda.to_cpu(pred_bbox)
                pred_label = chainer.cuda.to_cpu(pred_label)
                pred_score = chainer.cuda.to_cpu(pred_score)

                out_file = self.filename.format(
                    index=idx, iteration=trainer.updater.iteration)
                out_file = os.path.join(trainer.out, out_file)

                fig = plot.figure()

                ax_gt = fig.add_subplot(2, 1, 1)
                ax_gt.set_title('ground truth')
                vis_bbox(
                    img, gt_bbox, gt_label,
                    label_names=self.label_names, ax=ax_gt)

                ax_pred = fig.add_subplot(2, 1, 2)
                ax_pred.set_title('prediction')
                vis_bbox(
                    img, pred_bbox, pred_label, pred_score,
                    label_names=self.label_names, ax=ax_pred)

                plot.savefig(out_file)
                plot.close()

                idx += 1
