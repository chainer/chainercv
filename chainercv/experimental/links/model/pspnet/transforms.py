from __future__ import division

import math
import numpy as np


def convolution_crop(img, size, stride, return_param=False):
    """Convolution crop

    """
    _, H, W = img.shape

    hh = int(math.ceil((H - size[0]) / stride[0])) + 1
    ww = int(math.ceil((W - size[1]) / stride[1])) + 1

    start_y = -(size[0] + stride[0] * (hh - 1) - H) // 2
    start_x = -(size[1] + stride[1] * (ww - 1) - W) // 2

    crop_imgs = []
    y_slices = []
    x_slices = []
    crop_y_slices = []
    crop_x_slices = []
    for yy in range(hh):
        for xx in range(ww):
            y_min = yy * stride[0] + start_y
            x_min = xx * stride[1] + start_x
            y_max = y_min + size[0]
            x_max = x_min + size[1]

            crop_y_min = np.abs(np.minimum(y_min, 0))
            crop_x_min = np.abs(np.minimum(x_min, 0))
            crop_y_max = size[0] - np.maximum(y_max - H, 0)
            crop_x_max = size[1] - np.maximum(x_max - W, 0)

            crop_img = np.zeros((img.shape[0], size[0], size[1]),
                                dtype=img.dtype)
            y_slice = slice(max(y_min, 0), min(y_max, H))
            x_slice = slice(max(x_min, 0), min(x_max, W))
            crop_y_slice = slice(crop_y_min, crop_y_max)
            crop_x_slice = slice(crop_x_min, crop_x_max)
            crop_img[:, crop_y_slice, crop_x_slice] = img[:, y_slice, x_slice]

            crop_imgs.append(crop_img)
            y_slices.append(y_slice)
            x_slices.append(x_slice)
            crop_y_slices.append(crop_y_slice)
            crop_x_slices.append(crop_x_slice)

    if return_param:
        param = {'y_slices': y_slices, 'x_slices': x_slices,
                 'crop_y_slices': crop_y_slices,
                 'crop_x_slices': crop_x_slices}
        return np.array(crop_imgs), param
    else:
        return np.array(crop_imgs)
