import itertools
import numpy as np


def apply_semantic_segmentation_link(target, iterator, hook=None):
    """Apply a semantic segmentation link to an iterator

    This function applies a semantic segmentation link to an iterator.
    It stacks the outputs of the semantic segmentation link
    against :obj:`pred_labels`.
    This function also stacks the values returned by the iterator
    except the input image.
    These values can be used for evaluation.

    Args:
        target (chainer.Link): A semantic segmentation link. This link must
            have :meth:`predict` method which take a list of images and returns
            :obj:`labels`.
        iterator (chainer.Iterator): An iterator. Each sample should have
            an image as its first element. This image is passed to
            :meth:`target.predict` as an argument.
            The rests are stacked against :obj:`gt_values`.
        hook: A callable which is called after each iteration.
            :obj:`pred_labels` and :obj:`gt_values` are passed as arguments.
            Note that these values do not contain data from the previous
            iterations.

    Returns:
        An iterator and a tuple:
        This function returns :obj:`pred_labels` and :obj:`gt_values`.
        :obj:`gt_values` is a tuple of iterators. Each iterator corresponds
        to a value of a sample from the iterator.
        For example, if the iterator returns a batch of
        :obj:`(img, val0, val1)`, :obj:`next(gt_values)`
        will be :obj:`(val0, val1)`.
    """

    iterators = itertools.tee(_apply(target, iterator, hook))
    pred_labels = iterators[0]
    gt_values = iterators[1:]
    return pred_labels, gt_values


def _apply(target, iterator, hook):
    for batch in iterator:
        batch_imgs = list()
        batch_gt_values = list()

        for sample in batch:
            if isinstance(sample, np.ndarray):
                batch_imgs.append(sample)
                batch_gt_values.append(tuple())
            else:
                batch_imgs.append(sample[0])
                batch_gt_values.append(sample[1:])

        batch_pred_labels = target.predict(batch_imgs)

        if hook:
            hook(
                batch_pred_labels,
                tuple(list(bv) for bv in zip(*batch_gt_values)))

        for pred_label, gt_value in zip(batch_pred_labels, batch_gt_values):
            yield (pred_label,) + gt_value
