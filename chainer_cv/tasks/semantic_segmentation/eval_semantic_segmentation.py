from __future__ import division
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    """
    Returns
        numpy.ndarray of shape (n_class, n_class)
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_true, label_pred, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc

    Args:
        label_true (numpy.ndarray): a integer 2D image of classes. A pixel
            with value "-1" will be ignored during evaluation.
        label_pred (numpy.ndarray): a integer 2D image.
        n_class (int):  number of classes
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
