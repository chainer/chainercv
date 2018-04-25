import argparse
import numpy as np

import chainer
from chainer.links import Convolution2D
from chainer import serializers

from chainercv.links import Conv2DBNActiv
from chainercv.links import YOLOv3


def load_param(file, param):
    if isinstance(param, chainer.Variable):
        param = param.array
    param[:] = np.fromfile(file, dtype=np.float32, count=param.size) \
                 .reshape(param.shape)


def load_link(file, link):
    if isinstance(link, Convolution2D):
        load_param(file, link.b)
        load_param(file, link.W)
    elif isinstance(link, Conv2DBNActiv):
        load_param(file, link.bn.beta)
        load_param(file, link.bn.gamma)
        load_param(file, link.bn.avg_mean)
        load_param(file, link.bn.avg_var)
        load_param(file, link.conv.W)
    elif isinstance(link, chainer.ChainList):
        for l in link:
            load_link(file, l)


def load_yolo_v3(file, model):
    for i, link in enumerate(model.extractor):
        load_link(file, link)
        if i in {33, 39, 45}:
            subnet = model.subnet[(i - 33) // 6]
            load_link(file, subnet)
            # xy -> yx
            for data in (subnet[-1].W.array, subnet[-1].b.array):
                data = data.reshape(
                    (-1, 4 + 1 + model.n_fg_class) + data.shape[1:])
                data[:, [1, 0, 3, 2]] = data[:, :4].copy()


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

        load_yolo_v3(f, model)

    serializers.save_npz(args.output, model)


if __name__ == '__main__':
    main()
