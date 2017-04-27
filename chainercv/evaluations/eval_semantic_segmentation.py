from __future__ import division
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    # Construct histogram for label evaluation.

    mask = (label_true >= 0) & (label_true < n_class)
    # an array of shape (n_class, n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def eval_semantic_segmentation(label_pred, label_true, n_class):
    """Evaluate results of semantic segmentation.

    This function measures four metrics: pixel accuracy,
    mean pixel accuracy, mean intersection over union and
    frequency weighted intersection over union.

    The definition of these metrics are as follows, where
    :math:`p_{ij}` is the amount of pixels of class :math:`i`
    inferred to belong to :math:`j` and there is :math:`k` classes.

    * Pixel Accuracy (PA)
        :math:`PA = \\frac
        {\\sum_{i=1}^k p_{ii}}
        {\\sum_{i=1}^k \\sum_{j=1}^k p_{ij}}`
    * Mean Pixel Accuracy (MPA)
        :math:`MPA = \\frac{1}{k}
        \\sum_{i=1}^k
        \\frac{p_{ii}}{\\sum_{j=1}^k p_{ij}}`
    * Mean Intersection over Union (MIoU)
        :math:`MIoU = \\frac{1}{k}
        \\sum_{i=1}^k
        \\frac{p_{ii}}{\\sum_{j=1}^k p_{ij} + \\sum_{j=1}^k p_{ji} - p_{ii}}`
    * Frequency Weighted Intersection over Union (FWIoU)
        :math:`FWIoU = \\frac{1}{\\sum_{i=1}^k \\sum_{j=1}^k p_{ij}}
        \\sum_{i=1}^k \\frac{\\sum_{j=1}^k p_{ij}p_{ii}}
        {\\sum_{j=1}^k p_{ij} + \\sum_{j=1}^k p_{ji} - p_{ii}}`

    Args:
        label_pred (~numpy.ndarray): An integer array of image containing
            class labels as values, which is obtained from inference.
            This has shape :math:`(N, 1, H, W)` or :math:`(1, H, W)`,
            where :math:`N` is size of the batch, :math:`H` is the height
            and :math:`W` is the width.
        label_true (~numpy.ndarray): An integer array of image containing
            the ground truth class labels as values. A pixel with value
            "-1" will be ignored during evaluation. Its shape is similar
            to :obj:`label_pred`.
            Its image size is equal to that of :obj:`label_pred`.
            This should be a one channel CHW formatted image.
        n_class (int): Number of classes.

    Returns:
        (numpy.ndarray, numpy.ndarary, numpy.ndarray, numpy.ndarray):
        A tuple of pixel accuracy, mean pixel accuracy, MIoU and FWIoU.
        These arrays arrays have shape :math:`(N,)`, where :math:`N` is
        the number of images in the input.

    """
    ndim = label_pred.ndim
    if label_pred.ndim != label_true.ndim:
        raise ValueError(
            'Ground truth and predicted label map should have same number '
            'of dimensions.')
    if ndim == 3:
        label_pred = label_pred[None]
        label_true = label_true[None]
    elif ndim < 3:
        raise ValueError('Input images need to be at least three dimensional.')

    if label_pred.shape[1] != 1 or label_true.shape[1] != 1:
        raise ValueError('Channel sizes of inputs need to be one.')

    N = len(label_pred)
    acc = np.zeros((N,))
    acc_cls = np.zeros((N,))
    mean_iu = np.zeros((N,))
    fwavacc = np.zeros((N,))
    for i in range(len(label_pred)):
        hist = _fast_hist(
            label_true[i].flatten(), label_pred[i].flatten(), n_class)
        acc[i] = np.diag(hist).sum() / hist.sum()
        acc_cls_i = np.diag(hist) / hist.sum(axis=1)
        acc_cls[i] = np.nanmean(acc_cls_i)
        iu_denominator = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        iu = np.diag(hist) / iu_denominator
        mean_iu[i] = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc[i] = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
