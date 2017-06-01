from __future__ import division

import numpy as np


def calc_semantic_segmentation_confusion(pred_labels, gt_labels, n_class):
    """Collect confusion matrix.

    Args:
        pred_label (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
            For example, This is an array of shape :math:`(N, H, W)`.
        gt_label (iterable of numpy.ndarray): A collection of ground
            truth label. The shape of a ground truth label array is
            :math:`(H, W)`. The corresponding prediction label should
            have the same shape.
            A pixel with value "-1" will be ignored during evaluation.
        n_class (int): The number of classes.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of times
        a pixel with ground truth class :math:`i` is predicted
        to be class :math:`j`.

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
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of inputs should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shapes of inputs should be same.')

        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()
        mask = (gt_label >= 0) & (gt_label < n_class)
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape(n_class, n_class)
    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with given confusion matrix.

    The definition of Intersection over Union (IoU) is as follows.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number times
            the pixel with ground truth class :math:`i` is predicted
            to be class :math:`j`.

    Returns:
        numpy.ndarray:
        IoU calculated from the given confusion matrix. Its shapes is
        :math:`(n\_class)`.

    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou


def eval_semantic_segmentation_iou(pred_labels, gt_labels, n_class):
    """Evaluate Intersection over Union from labels.

    This function calculates Intersection over Union (IoU)
    for the task of semantic segmentation.

    The definition of IoU and a related metric, mean Intersection
    over Union (mIoU), are as follow, where
    :math:`N_{ij}` is the amount of pixels of class :math:`i`
    inferred to belong to :math:`j` and there is :math:`k` classes.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    mIoU can be computed by taking :obj:`numpy.nanmean` of the IoUs returned
    by this function.
    The more detailed descriptions on the above metric can be found at a
    review on semantic segmentation [#]_.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_label (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_label (iterable of numpy.ndarray): A collection of ground
            truth label. The shape of a ground truth label array is
            :math:`(H, W)`. The corresponding prediction label should
            have the same shape.
            A pixel with value "-1" will be ignored during evaluation.
        n_class (int): The number of classes.

    Returns:
        numpy.ndarray:
        An IoU computed as above for each class. Its shape is
        :math:`(n\_class,)`.

    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels, n_class)
    iou = calc_semantic_segmentation_iou(confusion)
    return iou
