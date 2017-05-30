import numpy as np

from chainercv.utils.iterator.split_iterator import split_iterator


def apply_detection_link(target, iterator, hook=None):
    """Apply a detection link to an iterator

    This function applies a detection link to an iterator.
    It stacks the outputs of the detection link
    against three itarators,
    :obj:`pred_bboxes`, :obj:`pred_labels` and :obj:`pred_scores`.
    This function also stacks the values returned by the iterator
    except the input image.
    These values can be used for evaluation.

    Args:
        target (chainer.Link): A detection link. This link must have
            :meth:`predict` method which take a list of images and returns
            :obj:`bboxes`, :obj:`labels` and :obj:`scores`.
        iterator (chainer.Iterator): An iterator. Each sample should have
            an image as its first element. This image is passed to
            :meth:`target.predict` as an argument.
            The rests are stacked against :obj:`gt_values`.
        hook: A callable which is called after each iteration.
            :obj:`pred_bboxes`, :obj:`pred_labels`, :obj:`pred_scores` and
            :obj:`gt_values` are passed as arguments.
            Note that these values do not contain data from the previous
            iterations.

    Returns:
        Three iterators and a tuple:
        This function returns :obj:`pred_bboxes`,
        :obj:`pred_labels`, :obj:`pred_scores` and :obj:`gt_values`.
        :obj:`gt_values` is a tuple of iterators. Each iterator corresponds
        to a value of a sample from the iterator.
        For example, if the iterator returns a batch of
        :obj:`(img, val0, val1)`, :obj:`next(gt_values)` will be
        :obj:`(val0, val1)`.
    """

    iterators = split_iterator(_apply(target, iterator, hook))
    pred_bboxes = iterators[0]
    pred_labels = iterators[1]
    pred_scores = iterators[2]
    gt_values = iterators[3:]
    return pred_bboxes, pred_labels, pred_scores, gt_values


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

        batch_pred_bboxes, batch_pred_labels, batch_pred_scores = \
            target.predict(batch_imgs)

        if hook:
            hook(
                batch_pred_bboxes, batch_pred_labels, batch_pred_scores,
                tuple(list(bv) for bv in zip(*batch_gt_values)))

        for pred_bbox, pred_label, pred_score, gt_value in zip(
                batch_pred_bboxes, batch_pred_labels, batch_pred_scores,
                batch_gt_values):
            yield (pred_bbox, pred_label, pred_score) + gt_value
