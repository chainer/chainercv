from chainercv.utils.iterator.split_iterator import split_iterator


def apply_prediction_link(target, iterator, hook=None):
    imgs, pred_values, gt_values = split_iterator(
        _apply(target, iterator, hook))

    # imgs: iter of [img] -> iter of img
    imgs = _flatten(imgs)

    # pred_values: iter of ([pred_val0], [pred_val1], ...)
    #    -> (iter of pred_val0, iter of pred_val1, ...)
    pred_values = tuple(map(_flatten, split_iterator(pred_values)))

    # gt_values: iter of ([gt_val0], [gt_val1], ...)
    #    -> (iter of gt_val0, iter of gt_val1, ...)
    gt_values = tuple(map(_flatten, split_iterator(gt_values)))

    return imgs, pred_values, gt_values


def _apply(target, iterator, hook):
    for batch in iterator:
        # batch: [(img, gt_val0, gt_val1, ...)] or [img]

        imgs = list()
        gt_values = list()
        for sample in batch:
            if isinstance(sample, tuple):
                imgs.append(sample[0])
                gt_values.append(sample[1:])
            else:
                imgs.append(sample)
                gt_values.append(tuple())

        # imgs: [img]

        # gt_values: [(gt_val0, gt_val1, ...)] -> ([gt_val0], [gt_val1], ...)
        gt_values = tuple(list(v) for v in zip(*gt_values))

        # pred_values: ([pred_val0], [pred_val1], ...) or [pred_val]
        pred_values = target.predict(imgs)
        if not isinstance(pred_values, tuple):
            # pred_values: [pred_val] -> ([pred_val0], [pred_val1], ...)
            pred_values = pred_values,

        if hook:
            hook(imgs, pred_values, gt_values)

        yield imgs, pred_values, gt_values


def _flatten(iterator):
    return (sample for batch in iterator for sample in batch)
