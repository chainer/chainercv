import numpy as np
import random
import six

from chainercv import transforms
from chainercv import utils


def _random_distort(img):
    import cv2

    cv_img = img[::-1].transpose(1, 2, 0).astype(np.uint8)

    def convert(cv_img, alpha=1, beta=0):
        cv_img = cv_img.astype(float) * alpha + beta
        cv_img[cv_img < 0] = 0
        cv_img[cv_img > 255] = 255
        return cv_img.astype(np.uint8)

    def brightness(cv_img):
        if random.randrange(2):
            return convert(cv_img, beta=random.uniform(-32, 32))
        else:
            return cv_img

    def contrast(cv_img):
        if random.randrange(2):
            return convert(cv_img, alpha=random.uniform(0.5, 1.5))
        else:
            return cv_img

    def saturation(cv_img):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 1] = convert(
                cv_img[:, :, 1], alpha=random.uniform(0.5, 1.5))
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    def hue(cv_img):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 0] = (
                cv_img[:, :, 0].astype(int) + random.randint(-18, 18)) % 180
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    cv_img = brightness(cv_img)

    if random.randrange(2):
        cv_img = contrast(cv_img)
        cv_img = saturation(cv_img)
        cv_img = hue(cv_img)
    else:
        cv_img = saturation(cv_img)
        cv_img = hue(cv_img)
        cv_img = contrast(cv_img)

    return cv_img.astype(np.float32).transpose(2, 0, 1)[::-1]


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
    img = _random_distort(img)

    if random.randrange(2):
        img, param = transforms.random_expand(
            img, fill=mean, return_param=True)
        bbox = transforms.translate_bbox(
            bbox, x_offset=param['x_offset'], y_offset=param['y_offset'])

    img, bbox, label = _random_crop(img, bbox, label)

    _, H, W = img.shape
    img = transforms.resize(img, (size, size))
    bbox = transforms.resize_bbox(bbox, (W, H), (size, size))

    img, params = transforms.random_flip(
        img, x_random=True, return_param=True)
    bbox = transforms.flip_bbox(
        bbox, (size, size), x_flip=params['x_flip'])

    return img, bbox, label
