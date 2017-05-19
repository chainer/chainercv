from chainercv.datasets import CamVidDataset
import numpy as np

d = CamVidDataset(mode='train')

# for img, lbl in d:
#     print(len(np.unique(lbl)))

import chainer
import chainer.functions as F
import chainer.links as L

x = np.array([[0, 1, 2]])
t = np.array([2])