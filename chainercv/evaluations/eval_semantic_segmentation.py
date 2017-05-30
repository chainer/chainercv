from __future__ import division

import numpy as np
import six


def calc_confusion_matrix(pred_labels, gt_labels, n_class):
    """Collect confusion matrix.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. This is a batch of labels whose shape is :math:`(N, H, W)`
            or a list containing :math:`N` labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label. We assume that there are
            :math:`N` labels.
        gt_labels (iterable of numpy.ndarray): A collection of the ground
            truth labels.
            It is organized similarly to :obj:`pred_labels`. A pixel with value
            "-1" will be ignored during evaluation.
        n_class (int): The number of classes.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its row corresponds to the class id of the
        ground truth and its column corresponds to the class id of the
        prediction. Its shape is :math:`(n\_class, n\_class)`.

    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    if (isinstance(pred_labels, np.ndarray) and pred_labels.ndim != 3
            or isinstance(gt_labels, np.ndarray) and gt_labels.ndim != 3):
        raise ValueError('If batch of arrays are given, they have '
                         'to have dimension 3')
    N = len(pred_labels)

    if len(pred_labels) != len(gt_labels):
        raise ValueError('Number of the predicted labels and the'
                         'ground truth labels are different')
    for i in six.moves.range(N):
        if pred_labels[i].shape != gt_labels[i].shape:
            raise ValueError('Shape of the prediction and'
                             'the ground truth should match')

    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for i in six.moves.range(N):
        pred_label = pred_labels[i].flatten()
        gt_label = gt_labels[i].flatten()
        mask = (gt_label >= 0) & (gt_label < n_class)
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape(n_class, n_class)
    return confusion


def eval_semantic_segmentation(confusion):
    """Evaluate results of semantic segmentation.

    This function calculates Intersection over Union (IoU).

    The definition of IoU and a related metric, mean Intersection
    over Union (mIoU), are as follows, where
    :math:`N_{ij}` is the amount of pixels of class :math:`i`
    inferred to belong to :math:`j` and there is :math:`k` classes.

    * :math:`\\text{IoU of i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    mIoU can be computed by taking :obj:`numpy.nanmean` of the IoUs calculated
    by this function.
    The more detailed descriptions on the above metric can be found at a
    review on semantic segmentation [#]_.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        confusion (numpy.ndarray): Confusion matrix calculated by
            :func:`chainercv.evaluations.calc_confusion_matrix`.
            Its shape is :math:`(n\_class, n\_class)`.

    Returns:
        numpy.ndarray:
        IoUs computed from the given confusion matrix.
        Its shape is :math:`(n\_class,)`.

    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou
