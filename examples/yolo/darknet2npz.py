import argparse
import numpy as np

import chainer
from chainer import serializers

from chainercv.links import Conv2DBNActiv
from chainercv.links import YOLOv3


def load(file, link):
    if isinstance(link, Conv2DBNActiv):
        for param in (
                link.bn.beta.array,
                link.bn.gamma.array,
                link.bn.avg_mean,
                link.bn.avg_var,
                link.conv.W.array):
            param[:] = np.fromfile(file, dtype=np.float32, count=param.size) \
                         .reshape(param.shape)
    elif isinstance(link, chainer.ChainList):
        for l in link:
            load(file, l)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_fg_class', type=int, default=80)
    parser.add_argument('darknetmodel')
    parser.add_argument('output')
    args = parser.parse_args()

    model = YOLOv3(args.n_fg_class)
    with chainer.using_config('train', False):
        model(np.empty((1, 3, model.insize, model.insize), dtype=np.float32))

    with open(args.darknetmodel, mode='rb') as f:
        major = np.fromfile(f, dtype=np.int32, count=1)
        minor = np.fromfile(f, dtype=np.int32, count=1)
        np.fromfile(f, dtype=np.int32, count=1)  # revision
        assert(major * 10 + minor >= 2 and major < 1000 and minor < 1000)
        np.fromfile(f, dtype=np.int64, count=1)  # seen

        load(f, model.extractor)

    serializers.save_npz(args.output, model)


if __name__ == '__main__':
    main()
