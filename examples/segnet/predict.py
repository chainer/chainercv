import argparse

from chainer import cuda
from chainer.dataset import concat_examples
import chainer.functions as F
from chainer import serializers

from chainercv.datasets import CamVidDataset
from chainercv.links.model import SegNetBasic

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--snapshot', type=str)
parser.add_argument('--batchsize', type=int, default=12)
args = parser.parse_args()

model = SegNetBasic(12)
model.train = False
serializers.load_npz(args.snapshot, model)
if args.gpu >= 0:
    model.to_gpu(args.gpu)

test = CamVidDataset(mode='test')

n_positive = [0 for _ in range(len())]
n_true = []
n_true_positive = 0
for i in range(0, len(test), args.batchsize):
    img, lbl = concat_examples(test[i:i + args.batchsize], args.gpu)
    y = F.argmax(F.softmax(model(img)), axis=1)
    y, t = cuda.to_cpu(y.data), cuda.to_cpu(lbl.data)
    break
