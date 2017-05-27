import argparse

from chainer import cuda
from chainer.dataset import concat_examples
from chainer import iterators
from chainer import serializers

from chainercv.datasets import CamVidDataset
from chainercv.links import SegNetBasic


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--snapshot', type=str)
args = parser.parse_args()

model = SegNetBasic(n_class=11)
serializers.load_npz(args.snapshot, model)
model.to_gpu(args.gpu)

d = CamVidDataset(split='train')
it = iterators.SerialIterator(d, 24, repeat=False, shuffle=False)
bn_params = {}
num_iterations = 0
for batch in it:
    imgs, labels = concat_examples(batch, device=args.gpu)
    model(imgs)
    for name, link in model.namedlinks():
        if name.endswith('_bn'):
            if name not in bn_params:
                bn_params[name] = [cuda.to_cpu(link.avg_mean),
                                   cuda.to_cpu(link.avg_var)]
            else:
                bn_params[name][0] += cuda.to_cpu(link.avg_mean)
                bn_params[name][1] += cuda.to_cpu(link.avg_var)
    num_iterations += 1

for name, params in bn_params.items():
    bn_params[name][0] /= num_iterations
    bn_params[name][1] /= num_iterations

for name, link in model.namedlinks():
    if name.endswith('_bn'):
        link.avg_mean = bn_params[name][0]
        link.avg_var = bn_params[name][1]

serializers.save_npz('{}_infer'.format(args.snapshot), model)
