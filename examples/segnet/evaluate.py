import argparse

import chainer
from chainer import cuda
from chainer.dataset import concat_examples
import chainer.functions as F
from chainer import serializers
import numpy as np
from chainercv.datasets import CamVidDataset
from chainercv.links.model import SegNetBasic
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--snapshot', type=str)
parser.add_argument('--batchsize', type=int, default=24)
parser.add_argument('--n_class', type=int, default=11)
parser.add_argument('--ignore_labels', type=int, nargs='*', default=[11])
args = parser.parse_args()

model = SegNetBasic(args.n_class)
model.train = False
serializers.load_npz(args.snapshot, model)
if args.gpu >= 0:
    model.to_gpu(args.gpu)

test = CamVidDataset(mode='test')

n_positive = [0 for _ in range(args.n_class)]
n_true = [0 for _ in range(args.n_class)]
n_true_positive = [0 for _ in range(args.n_class)]
for i in tqdm(range(0, len(test), args.batchsize)):
    img, lbl = concat_examples(test[i:i + args.batchsize], args.gpu)
    img = chainer.Variable(img, volatile=True)
    y = F.argmax(F.softmax(model(img)), axis=1)
    y, t = cuda.to_cpu(y.data), cuda.to_cpu(lbl)
    for cls_i in range(args.n_class):
        if cls_i in args.ignore_labels:
            continue
        n_positive[cls_i] += np.sum(y == cls_i)
        n_true[cls_i] += np.sum(t == cls_i)
        n_true_positive[cls_i] += np.sum((y == cls_i) *  (t == cls_i))

ious = []
for cls_i in range(args.n_class):
    if cls_i in args.ignore_labels:
        continue
    iou = n_true_positive[cls_i] / float(n_positive[cls_i] + n_true[cls_i] - n_true_positive[cls_i])
    ious.append(iou)
    print('{}:'.format(cls_i), iou)
print('mean:', np.mean(ious))
