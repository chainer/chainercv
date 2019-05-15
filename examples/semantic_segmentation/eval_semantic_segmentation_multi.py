import argparse

import chainer
from chainer import iterators

import chainermn

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from eval_semantic_segmentation import models
from eval_semantic_segmentation import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=('cityscapes', 'ade20k', 'camvid', 'voc'))
    parser.add_argument('--model', choices=sorted(models.keys()))
    parser.add_argument('--pretrained-model')
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--input-size', type=int, default=None)
    args = parser.parse_args()

    comm = chainermn.create_communicator('pure_nccl')
    device = comm.intra_rank

    dataset, eval_, model, batchsize = setup(
        args.dataset, args.model, args.pretrained_model,
        args.batchsize, args.input_size)

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    model.use_preset('evaluate')

    if not comm.rank == 0:
        apply_to_iterator(model.predict, None, comm=comm)
        return

    it = iterators.MultithreadIterator(
        dataset, batchsize * comm.size, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, it, hook=ProgressHook(len(dataset)), comm=comm)
    # Delete an iterator of images to save memory usage.
    del in_values

    eval_(out_values, rest_values)


if __name__ == '__main__':
    main()
