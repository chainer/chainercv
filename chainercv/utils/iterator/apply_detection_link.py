import numpy as np


def apply_detection_link(target, iterator, hook=None):
    """Apply a detection link to an iterator

    This function applies a detection link to an iterator.
    It stacks the outputs of the detection link
    into three lists,
    :obj:`pred_bboxes`, :obj:`pred_labels` and :obj:`pred_scores`.
    This function also stacks the values returned by the iterator.
    These values can be used for evaluation.

    Args:
        target (chainer.Link): An detection link. This link must have
            :meth:`predict` method which take a list of images and returns
            :obj:`bboxes`, :obj:`labels` and :obj:`scores`.
        iterator (chainer.Iterator): An iterator. Each sample should have
            an image as its first element. This image is passed to
            :obj:`target`. The rests are stacked into :obj:`gt_values`.
        hook: An callable which is called after each iteration.
            :obj:`pred_bboxes`, :obj:`pred_labels`, :obj:`pred_scores` and
            :obj:`gt_values` are passed as arguments.
            Note that these values do not contain data from the previous
            iterations

    Returns:
        tuple of lists:
        This function returns four lists: :obj:`pred_bboxes`,
        :obj:`pred_labels`, :obj:`pred_scores` and :obj:`gt_values`.
        :obj:`gt_values` is a tuple of lists. Each list corresponds to an value
        of samples from the iterator. For example, if the iterator returns
        batches of :obj:`img, val0, val1`, :obj:`gt_values` will be
        :obj:`([val0], [val1])`.
    """

    pred_bboxes = list()
    pred_labels = list()
    pred_scores = list()
    gt_values = None

    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        batch_imgs = list()
        batch_gt_values = list()

        for sample in batch:
            if isinstance(sample, np.ndarray):
                batch_imgs.append(sample)
                batch_gt_values.append(tuple())
            else:
                batch_imgs.append(sample[0])
                batch_gt_values.append(sample[1:])

        batch_gt_values = tuple(list(bv) for bv in zip(*batch_gt_values))

        batch_pred_bboxes, batch_pred_labels, batch_pred_scores = \
            target.predict(batch_imgs)

        if hook:
            hook(
                batch_pred_bboxes, batch_pred_labels, batch_pred_scores,
                batch_gt_values)

        pred_bboxes.extend(batch_pred_bboxes)
        pred_labels.extend(batch_pred_labels)
        pred_scores.extend(batch_pred_scores)

        if gt_values is None:
            gt_values = batch_gt_values
        else:
            for v, bv in zip(gt_values, batch_gt_values):
                v.extend(bv)

    return pred_bboxes, pred_labels, pred_scores, gt_values
