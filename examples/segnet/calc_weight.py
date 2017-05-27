from __future__ import division

import numpy as np

from chainercv.datasets import CamVidDataset


n_class = 12
dataset = CamVidDataset(split='train')

n_cls_pixels = [0 for _ in range(n_class)]
n_img_pixels = [0 for _ in range(n_class)]

for img, lbl in dataset:
    for cls_i in np.unique(lbl):
        n_cls_pixels[cls_i] += np.sum(lbl == cls_i)
        n_img_pixels[cls_i] += lbl.size
freq = []
for n_cls_pixel, n_img_pixel in zip(n_cls_pixels, n_img_pixels):
    freq.append(n_cls_pixel / n_img_pixel)
freq = np.array(freq)
median_freq = np.median(freq)

np.save('class_weight', median_freq / freq)
