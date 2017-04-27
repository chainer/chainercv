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
            This should be a one channel CHW formatted image.
        label_true (~numpy.ndarray): An integer array of image containing
            the ground truth class labels as values. A pixel with value
            "-1" will be ignored during evaluation.
            Its image size is equal to that of :obj:`label_pred`.
            This should be a one channel CHW formatted image.
        n_class (int): Number of classes.

    Returns:
        (float, float, float, float):
        A tuple of pixel accuracy, mean pixel accuracy, MIoU and FWIoU.

    """
    hist = _fast_hist(label_true.flatten(), label_pred.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
