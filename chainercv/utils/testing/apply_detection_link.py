import numpy as np


def apply_detection_link(target, iterator, hook=None):
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

        batch_gt_values = [list(bv) for bv in zip(*batch_gt_values)]

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
