import numpy as np
import random
import six

from chainercv import transforms
from chainercv import utils


def _random_crop(img, bbox, label):
    if len(bbox) == 0:
        return img, bbox, label

    constraints = (
        (0.1, None),
        (0.3, None),
        (0.5, None),
        (0.7, None),
        (0.9, None),
        (None, 1),
    )

    max_trial = 50

    _, H, W = img.shape

    crop_bbox = [np.array((0, 0, W, H))]
    for iou_min, iou_max in constraints:
        for _ in six.moves.range(max_trial):
            if iou_min is None:
                iou_min = 0
            if iou_max is None:
                iou_max = 1

            scale = random.uniform(0.3, 1)
            aspect_ratio = random.uniform(
                max(1 / 2, scale * scale), min(2, 1 / (scale * scale)))
            crop_w = int(W * scale * np.sqrt(aspect_ratio))
            crop_h = int(H * scale / np.sqrt(aspect_ratio))

            crop_l = random.randrange(W - crop_w)
            crop_t = random.randrange(H - crop_h)
            crop_bb = np.array((
                crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            iou = utils.bbox_iou(bbox, crop_bb[np.newaxis])
            if iou_min <= iou.min() and iou.max() <= iou_max:
                crop_bbox.append(crop_bb)
                break

    crop_bb = random.choice(crop_bbox)

    img = img[:, crop_bb[1]:crop_bb[3], crop_bb[0]:crop_bb[2]]

    center = (bbox[:, :2] + bbox[:, 2:]) / 2
    mask = np.logical_and(crop_bb[:2] < center, center < crop_bb[2:]) \
             .all(axis=1)
    bbox = bbox[mask].copy()
    label = label[mask]

    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, 2:] -= crop_bb[:2]

    return img, bbox, label


def random_transform(img, bbox, label, size, mean):
    # color augmentations here

    if random.randrange(2):
        img, param = transforms.random_expand(
            img, fill=mean, return_param=True)
        bbox = transforms.translate_bbox(
            bbox, param['x_offset'], param['y_offset'])

    img, bbox, label = _random_crop(img, bbox, label)

    _, H, W = img.shape
    img = transforms.resize(img, (size, size))
    bbox = transforms.resize_bbox(bbox, (W, H), (size, size))

    img, params = transforms.random_flip(
        img, x_random=True, return_param=True)
    bbox = transforms.flip_bbox(
        bbox, (size, size), x_flip=params['x_flip'])

    return img, bbox, label
