
import copy
import numpy as np
import os.path as osp
from skimage.io import imread
from skimage.color import gray2rgb
import tarfile

import chainer
from chainer.dataset import download

from chainer_cv import utils
from chainer_cv.wrappers import KeepSubsetWrapper


root = 'pfnet/chainer_cv/imagenet'

def _get_imagenet(urls):

    data_root = download.get_dataset_directory(root)
    #base_path = osp.join(data_root, 'Stanford_Online_Products')
    #if osp.exists(base_path):
    #    return base_path

    for url in urls:
        download_file_path = utils.cached_download(url)

        with tarfile.TarFile(download_file_path, 'r') as t:
            t.extractall(data_root)
    return base_path



class ImagenetDataset(chainer.datasets.ImageDataset):

    def __init__(self, base_dir='auto', urls=None):
        if base_dir == 'auto':
            base_dir = _get_imagenet(urls)
        self.base_dir = base_dir


if __name__ == '__main__':
    urls = ''
    ImagenetDataset(urls=urls)
