from __future__ import division

import numpy as np


def calc_confusion_matrix(pred_label, gt_label, n_class):
    """Collect confusion matrix.

    Args:
        pred_label (numpy.ndarray): A predicted label.
            The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_label (numpy.ndarray): The ground truth label.
            Its shape is :math:`(H, W)`.
            A pixel with value "-1" will be ignored during evaluation.
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
    if pred_label.ndim != 2 or gt_label.ndim != 2:
        raise ValueError('ndim of inputs should be two.')
    if pred_label.shape != gt_label.shape:
        raise ValueError('Shapes of inputs should be same.')

    pred_label = pred_label.flatten()
    gt_label = gt_label.flatten()
    mask = (gt_label >= 0) & (gt_label < n_class)
    confusion = np.bincount(
        n_class * gt_label[mask].astype(int) +
        pred_label[mask], minlength=n_class**2).reshape(n_class, n_class)
    return confusion


def eval_semantic_segmentation(pred_labels, gt_labels, n_class):
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
        pred_labels (iterator or iterable of numpy.ndarray):
        gt_labels (iterator or iterable of numpy.ndarray):
        n_class (int): The number of classes.

    Returns:
        numpy.ndarray:
        IoUs computed from the given confusion matrix.
        Its shape is :math:`(n\_class,)`.

    """
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    while True:
        try:
            pred_label = next(pred_labels)
            gt_label = next(gt_labels)
        except StopIteration:
            break
        confusion += calc_confusion_matrix(pred_label, gt_label, n_class)

    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou
