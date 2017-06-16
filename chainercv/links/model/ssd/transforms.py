from __future__ import division

import numpy as np
import random
import six

from chainercv import utils


def random_distort(img):
    import cv2

    cv_img = img[::-1].transpose(1, 2, 0).astype(np.uint8)

    def convert(img, alpha=1, beta=0):
        img = img.astype(float) * alpha + beta
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

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


def random_crop_with_bbox(
        img, bbox, min_scale=0.3, max_scale=1,
        max_aspect_ratio=2, constraints=None,
        max_trial=50, return_param=False):

    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    _, H, W = img.shape
    params = [{
        'constraint': None, 'scale': 1, 'aspect_ratio': 1,
        'y_slice': slice(0, H), 'x_slice': slice(0, W)}]

    if len(bbox) == 0:
        constraints = list()

    for iou_min, iou_max in constraints:
        if iou_min is None:
            iou_min = 0
        if iou_max is None:
            iou_max = 1

        for _ in six.moves.range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(H * scale / np.sqrt(aspect_ratio))
            crop_w = int(W * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(H - crop_h)
            crop_l = random.randrange(W - crop_w)
            crop_bb = np.array((
                crop_t, crop_l, crop_t + crop_h, crop_l + crop_w))

            iou = utils.bbox_iou(bbox, crop_bb[np.newaxis])
            if iou_min <= iou.min() and iou.max() <= iou_max:
                params.append({
                    'constraint': (iou_min, iou_max),
                    'scale': scale, 'aspect_ratio': aspect_ratio,
                    'y_slice': slice(crop_t, crop_t + crop_h),
                    'x_slice': slice(crop_l, crop_l + crop_w)})
                break

    param = random.choice(params)
    img = img[:, param['y_slice'], param['x_slice']]

    if return_param:
        return img, param
    else:
        return img


def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        contain_center_only=False, return_param=False):

    def parse_slice(slice_):
        if slice_ is None:
            return -np.inf, np.inf

        if slice_.start is None:
            l = -np.inf
        else:
            l = slice_.start
        if slice_.stop is None:
            u = np.inf
        else:
            u = slice_.stop
        return l, u

    t, b = parse_slice(y_slice)
    l, r = parse_slice(x_slice)
    crop_bb = np.array((t, l, b, r))

    if contain_center_only:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        mask = np.logical_and(crop_bb[:2] < center, center < crop_bb[2:]) \
                 .all(axis=1)
    else:
        mask = np.ones(bbox.shape[0], dtype=bool)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'mask': mask}
    else:
        return bbox


def resize_with_random_interpolation(img, size, return_param=False):
    import cv2

    cv_img = img[::-1].transpose(1, 2, 0)

    inters = (
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    )
    inter = random.choice(inters)
    cv_img = cv2.resize(cv_img, size, interpolation=inter)

    img = cv_img.astype(np.float32).transpose(2, 0, 1)[::-1]

    if return_param:
        return img, {'interpolation': inter}
    else:
        return img
