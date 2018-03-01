from chainercv.utils.iterator.unzip import unzip


def apply_to_iterator(func, iterator, n_input=1, hook=None):
    """Apply a function/method to batches from an iterator.

    This function applies a function/method to an iterator of batches.

    It assumes that the iterator iterates over a collection of tuples
    that contain inputs to :func:`func`.
    Additionally, the tuples may contain values
    that are not used by :func:`func`.
    For convenience, we allow the iterator to iterate over a collection of
    inputs that are not tuple.
    Here is an illustration of the expected behavior of the iterator.
    This behaviour is the same as :class:`chainer.Iterator`.

    >>> batch = next(iterator)
    >>> # batch: [in_val]
    or
    >>> # batch: [(in_val0, ..., in_val{n_input - 1})]
    or
    >>> # batch: [(in_val0, ..., in_val{n_input - 1}, rest_val0, ...)]

    :func:`func` should take batch(es) of data and
    return batch(es) of computed values.
    Here is an illustration of the expected behavior of the function.

    >>> out_vals = func([in_val0], ..., [in_val{n_input - 1}])
    >>> # out_vals: [out_val]
    or
    >>> out_vals0, out_vals1, ... = func([in_val0], ..., [in_val{n_input - 1}])
    >>> # out_vals0: [out_val0]
    >>> # out_vals1: [out_val1]

    With :func:`apply_to_iterator`, users can get iterator(s) of values
    returned by :func:`func`. It also returns iterator(s) of input values and
    values that are not used for computation.

    >>> in_values, out_values, rest_values = apply_to_iterator(
    >>>     func, iterator, n_input)
    >>> # in_values: (iter of in_val0, ..., iter of in_val{n_input - 1})
    >>> # out_values: (iter of out_val0, ...)
    >>> # rest_values: (iter of rest_val0, ...)

    Here is an exmple, which applies a pretrained Faster R-CNN to
    PASCAL VOC dataset.

    >>> from chainer import iterators
    >>>
    >>> from chainercv.datasets import VOCBBoxDataset
    >>> from chainercv.links import FasterRCNNVGG16
    >>> from chainercv.utils import apply_to_iterator
    >>>
    >>> dataset = VOCBBoxDataset(year='2007', split='test')
    >>> # next(iterator) -> [(img, gt_bbox, gt_label)]
    >>> iterator = iterators.SerialIterator(
    ...     dataset, 2, repeat=False, shuffle=False)
    >>>
    >>> # model.predict([img]) -> ([pred_bbox], [pred_label], [pred_score])
    >>> model = FasterRCNNVGG16(pretrained_model='voc07')
    >>>
    >>> in_values, out_values, rest_values = apply_to_iterator(
    ...     model.predict, iterator)
    >>>
    >>> # in_values contains one iterator
    >>> imgs, = in_values
    >>> # out_values contains three iterators
    >>> pred_bboxes, pred_labels, pred_scores = out_values
    >>> # rest_values contains two iterators
    >>> gt_bboxes, gt_labels = rest_values

    Args:
        func: A callable that takes batch(es) of input data and returns
            computed data.
        iterator (iterator): An iterator of batches.
            The first :obj:`n_input` elements in each sample are
            treated as input values. They are passed to :obj:`func`.
        n_input (int): The number of input data. The default value is :obj:`1`.
        hook: A callable that is called after each iteration.
            :obj:`in_values`, :obj:`out_values`, and :obj:`rest_values`
            are passed as arguments.
            Note that these values do not contain data from the previous
            iterations.

    Returns:
        Three tuples of iterators:
        This function returns three tuples of iterators:
        :obj:`in_values`, :obj:`out_values` and :obj:`rest_values`.

        * :obj:`in_values`: A tuple of iterators. Each iterator \
            returns a corresponding input value. \
            For example, if :func:`func` takes \
            :obj:`[in_val0], [in_val1]`, :obj:`next(in_values[0])` \
            and :obj:`next(in_values[1])` will be \
            :obj:`in_val0` and :obj:`in_val1`.
        * :obj:`out_values`: A tuple of iterators. Each iterator \
            returns a corresponding computed value. \
            For example, if :func:`func` returns \
            :obj:`([out_val0], [out_val1])`, :obj:`next(out_values[0])` \
            and :obj:`next(out_values[1])` will be \
            :obj:`out_val0` and :obj:`out_val1`.
        * :obj:`rest_values`: A tuple of iterators. Each iterator \
            returns a corresponding rest value. \
            For example, if the :obj:`iterator` returns \
            :obj:`[(in_val0, in_val1, rest_val0, rest_val1)]`, \
            :obj:`next(rest_values[0])` \
            and :obj:`next(rest_values[1])` will be \
            :obj:`rest_val0` and :obj:`rest_val1`. \
            If the input \
            iterator does not give any rest values, this tuple \
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

        in_values = []
        rest_values = []
        for sample in batch:
            if isinstance(sample, tuple):
                in_values.append(sample[0:n_input])
                rest_values.append(sample[n_input:])
            else:
                in_values.append((sample,))
                rest_values.append(())

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
