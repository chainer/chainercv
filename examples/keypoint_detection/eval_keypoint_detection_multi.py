import argparse

import chainer
from chainer import iterators
import chainermn

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from eval_keypoint_detection import models
from eval_keypoint_detection import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=('coco',), default='coco')
    parser.add_argument('--model', choices=sorted(models.keys()))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--batchsize', type=int)
    args = parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    dataset, label_names, eval_, model, batchsize = setup(
        args.dataset, args.model, args.pretrained_model, args.batchsize)

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    if not comm.rank == 0:
        apply_to_iterator(model.predict, None, comm=comm)
        return

    iterator = iterators.MultithreadIterator(
        dataset, batchsize * comm.size, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)), comm=comm)
    # delete unused iterators explicitly
    del in_values

    eval_(out_values, rest_values)


if __name__ == '__main__':
    main()
