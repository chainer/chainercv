from chainercv.utils.iterator.unzip import unzip


def apply_prediction_to_iterator(func, iterator, n_input=1, hook=None):
    """Apply a prediction function/method to an iterator.

    This function applies a prediction function/method to an iterator.
    It assumes that the iterator returns a batch of images or
    a batch of tuples whose first element is an image. In the case that
    it returns a batch of tuples, the rests are treated as ground truth
    values.

    >>> imgs = next(iterator)
    >>> # imgs: [img]
    or
    >>> batch = next(iterator)
    >>> # batch: [(img, gt_val0, gt_val1)]

    This function applys :func:`predict` to a batch of images and gets
    predicted value(s). :func:`predict` should take a batch of images and
    return a batch of prediction values
    or a tuple of batches of prediction values.

    >>> pred_vals0 = predict(imgs)
    >>> # pred_vals0: [pred_val0]
    or
    >>> pred_vals0, pred_vals1 = predict(imgs)
    >>> # pred_vals0: [pred_val0]
    >>> # pred_vals1: [pred_val1]

    Here is an exmple, which applies a pretrained Faster R-CNN to
    PASCAL VOC dataset.

    >>> from chainer import iterators
    >>>
    >>> from chainercv.datasets import VOCDetectionDataset
    >>> from chainercv.links import FasterRCNNVGG16
    >>> from chainercv.utils import apply_prediction_to_iterator
    >>>
    >>> dataset = VOCDetectionDataset(year='2007', split='test')
    >>> # next(iterator) -> [(img, gt_bbox, gt_label)]
    >>> iterator = iterators.SerialIterator(
    ...     dataset, 2, repeat=False, shuffle=False)
    >>>
    >>> # model.predict([img]) -> ([pred_bbox], [pred_label], [pred_score])
    >>> model = FasterRCNNVGG16(pretrained_model='voc07')
    >>>
    >>> imgs, pred_values, gt_values = apply_prediction_to_iterator(
    ...     model.predict, iterator)
    >>>
    >>> # pred_values contains three iterators
    >>> pred_bboxes, pred_labels, pred_scores = pred_values
    >>> # gt_values contains two iterators
    >>> gt_bboxes, gt_labels = gt_values

    Args:
        predict: A callable that takes a batch of images and returns
            prediction.
        iterator (chainer.Iterator): An iterator. Each sample should have
            an image as its first element. This image is passed to
            :func:`predict` as an argument.
            The rests are treated as ground truth values.
        hook: A callable that is called after each iteration.
            :obj:`imgs`, :obj:`pred_values` and :obj:`gt_values` are passed as
            arguments.
            Note that these values do not contain data from the previous
            iterations.

    Returns:
        An iterator and two tuples of iterators:
        This function returns an iterator and two tuples of iterators:
        :obj:`imgs`, :obj:`pred_values` and :obj:`gt_values`.

        * :obj:`imgs`: An iterator that returns an image.
        * :obj:`pred_values`: A tuple of iterators. Each iterator \
            returns a corresponding predicted value. \
            For example, if :func:`predict` returns \
            :obj:`([pred_val0], [pred_val1])`, :obj:`next(pred_values[0])` \
            and :obj:`next(pred_values[1])` will be \
            :obj:`pred_val0` and :obj:`pred_val1`.
        * :obj:`gt_values`: A tuple of iterators. Each iterator \
            returns a corresponding ground truth value. \
            For example, if the :obj:`iterator` returns \
            :obj:`[(img, gt_val0, gt_val1)]`, :obj:`next(gt_values[0])` \
            and :obj:`next(gt_values[1])` will be \
            :obj:`gt_val0` and :obj:`gt_val1`. \
            If the input \
            iterator does not give any ground truth values, this tuple \
            will be empty.
    """

    in_values, out_values, rest_values = unzip(
        _apply(func, iterator, n_input, hook))

    # in_values: iter of ([in_val0], [in_val1], ...)
    #     -> (iter of in_val0, iter of in_val1, ...)
    in_values = tuple(map(_flatten, unzip(in_values)))

    # out_values: iter of ([out_val0], [out_val1], ...)
    #     -> (iter of out_val0, iter of out_val1, ...)
    out_values = tuple(map(_flatten, unzip(out_values)))

    # rest_values: iter of ([rest_val0], [rest_val1], ...)
    #     -> (iter of rest_val0, iter of rest_val1, ...)
    rest_values = tuple(map(_flatten, unzip(rest_values)))

    return in_values, out_values, rest_values


def _apply(func, iterator, n_input, hook):
    for batch in iterator:
        # batch: [(in_val0, in_val1, ... , rest_val0, rest_val1, ...)] or
        #     [in_val]

        in_values = list()
        rest_values = list()
        for sample in batch:
            if isinstance(sample, tuple):
                in_values.append(sample[0:n_input])
                rest_values.append(sample[n_input:])
            else:
                in_values.append((sample,))
                rest_values.append(tuple())

        # in_values: [(in_val0, in_val1, ...)]
        #     ->  ([in_val0], [in_val1], ...)
        in_values = tuple(list(v) for v in zip(*in_values))

        # rest_values: [(rest_val0, rest_val1, ...)]
        #     -> ([rest_val0], [rest_val1], ...)
        rest_values = tuple(list(v) for v in zip(*rest_values))

        # out_values: ([out_val0], [out_val1], ...) or [out_val]
        out_values = func(*in_values)
        if not isinstance(out_values, tuple):
            # pred_values: [out_val] -> ([out_val],)
            out_values = out_values,

        if hook:
            hook(in_values, out_values, rest_values)

        yield in_values, out_values, rest_values


def _flatten(iterator):
    return (sample for batch in iterator for sample in batch)
