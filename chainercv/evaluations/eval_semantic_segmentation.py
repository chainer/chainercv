from __future__ import division

import numpy as np
import six

from chainer import cuda


def _to_cpu(arrays, xp):
    if isinstance(arrays, xp.ndarray):
        out_arrays = cuda.to_cpu(arrays)
    else:
        out_arrays = []
        for array in arrays:
            out_arrays.append(cuda.to_cpu(array))
    return out_arrays


def _fast_hist(pred_label, gt_label, n_class):
    # Construct histogram for label evaluation.

    mask = (gt_label >= 0) & (gt_label < n_class)
    # an array of shape (n_class, n_class)
    hist = np.bincount(
        n_class * gt_label[mask].astype(int) +
        pred_label[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def eval_semantic_segmentation(pred_labels, gt_labels, n_class):
    """Evaluate results of semantic segmentation.

    This function measures four metrics: pixel accuracy,
    mean pixel accuracy, mean intersection over union and
    frequency weighted intersection over union.

    The definition of these metrics are as follows, where
    :math:`N_{ij}` is the amount of pixels of class :math:`i`
    inferred to belong to :math:`j` and there is :math:`k` classes.

    * Pixel Accuracy (PA)
        :math:`PA = \\frac
        {\\sum_{i=1}^k N_{ii}}
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * Mean Pixel Accuracy (mPA)
        :math:`mPA = \\frac{1}{k}
        \\sum_{i=1}^k
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * Mean Intersection over Union (mIoU)
        :math:`mIoU = \\frac{1}{k}
        \\sum_{i=1}^k
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * Frequency Weighted Intersection over Union (fwIoU)
        :math:`fwIoU = \\frac{1}{\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}
        \\sum_{i=1}^k \\frac{\\sum_{j=1}^k N_{ij}N_{ii}}
        {\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    The more detailed descriptions on the above metrics can be found at a
    review on semantic segmentation [#]_.

    Types of :obj:`pred_labels` and :obj:`gt_labels` need to be same.
    The outputs are same type as the inputs.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_labels (iterable of arrays): A collection of predicted
            labels. This is a batch of labels whose shape is :math:`(N, H, W)`
            or a list containing :math:`N` labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label. We assume that there are
            :math:`N` labels.
        gt_labels (iterable of arrays): A collection of the ground
            truth labels.
            It is organized similarly to :obj:`pred_labels`. A pixel with value
            "-1" will be ignored during evaluation.
        n_class (int): Number of classes.

    Returns:
        (array, array, array, array):
        A tuple of pixel accuracy, mean pixel accuracy, mIoU and fwIoU.
        These arrays have shape :math:`(N,)`, where :math:`N` is
        the number of images in the input.

    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    xp = cuda.get_array_module(pred_labels[0], gt_labels[0])
    if (isinstance(pred_labels, xp.ndarray) and pred_labels.ndim != 3
            or isinstance(gt_labels, xp.ndarray) and gt_labels.ndim != 3):
        raise ValueError('If batch of arrays are given, they have '
                         'to have dimension 3')
    pred_labels = _to_cpu(pred_labels, xp)
    gt_labels = _to_cpu(gt_labels, xp)
    N = len(pred_labels)

    if len(pred_labels) != len(gt_labels):
        raise ValueError('Number of the predicted labels and the'
                         'ground truth labels are different')
    for i in six.moves.range(N):
        if pred_labels[i].shape != gt_labels[i].shape:
            raise ValueError('Shape of the prediction and'
                             'the ground truth should match')

    acc = np.zeros((N,))
    acc_cls = np.zeros((N,))
    miou = np.zeros((N,))
    fwavacc = np.zeros((N,))
    for i in six.moves.range(N):
        hist = _fast_hist(
            pred_labels[i].flatten(), gt_labels[i].flatten(), n_class)
        acc[i] = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls_i = np.diag(hist) / hist.sum(axis=1)
            acc_cls[i] = np.nanmean(acc_cls_i)
        iou_denominator = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        iou = np.diag(hist) / iou_denominator
        miou[i] = np.nanmean(iou)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc[i] = (freq[freq > 0] * iou[freq > 0]).sum()

    return (xp.asarray(acc), xp.asarray(acc_cls),
            xp.asarray(miou), xp.asarray(fwavacc))
