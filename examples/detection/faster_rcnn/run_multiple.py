from train_faster_rcnn import main as train_main
from chainer_tools.tools.get_value_from_log import get_value_from_log
import os
import fire
import pickle
import numpy as np


experiments_txt = 'result/experiments.txt'

def do_analyze():
    results = []
    for l in open(experiments_txt):
        fn = os.path.join(l, 'log')
        if os.path.exists(fn):
            results.append(
                get_value_from_log(fn, 'main/map'))
    print results


def main(gpus=0, n=1, roi_batchsize=128, model_mode='vgg', analyze=False):
    if analyze:
        do_analyze()
    seeds = np.random.randint(0, 10000, size=(n,))
    for seed in seeds:
        out_dir = 'result/seed_{}'.format(seed)
        with open(experiments_txt, 'a') as f:
            f.write('{} \n'.format(out_dir))
        train_main(
            out=out_dir, seed=seed, gpus=gpus,
            model_mode=model_mode, roi_batchsize=roi_batchsize)


if __name__ == '__main__':
    fire.Fire(main)
