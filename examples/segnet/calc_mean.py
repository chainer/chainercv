from chainercv.datasets import CamVidDataset
import numpy as np

d = CamVidDataset(mode='train')

mean_img = None
for img, lbl in d:
    if mean_img is None:
        mean_img = img
    else:
        mean_img += img
mean_img /= len(d)

std_img = None
for img, lbl in d:
    if std_img is None:
        std_img = (img - mean_img) ** 2
    else:
        std_img += img
std_img /= len(d)

np.save('mean', mean_img)
np.save('std', std_img)