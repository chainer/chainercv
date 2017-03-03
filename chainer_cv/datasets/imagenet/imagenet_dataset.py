import os
import os.path as osp
import pickle
import tarfile

import chainer
from chainer.dataset import download

from chainer_cv import utils
from chainer_cv.datasets.utils import cache_load


root = 'yuyu2172/chainer-cv/imagenet'


def _get_imagenet(urls):
    data_root = download.get_dataset_directory(root)
    # this is error prone
    if osp.exists(osp.join(data_root, 'train')):
        return data_root

    for key, url in urls.items():
        download_file_path = utils.cached_download(url)

        d = osp.join(data_root, key)
        if not osp.exists(d):
            os.makedirs(d)
        with tarfile.TarFile(download_file_path, 'r') as t:
            t.extractall(d)

    # this is an extra step needed for train dataset
    train_dir = osp.join(data_root, 'train')
    for tar_fn in os.listdir(train_dir):
        if tar_fn[-3:] == 'tar':
            with tarfile.TarFile(osp.join(train_dir, tar_fn), 'r') as t:
                t.extractall(train_dir)
    return data_root


def get_imagenet_synset_map():
    synset_map = {}
    for l in open(osp.join(
            osp.split(osp.abspath(__file__))[0], 'synset_list.txt')):
        words = l.split()
        synset_map[words[0]] = words[1:]
    return synset_map


class ImagenetDataset(chainer.datasets.ImageDataset):
    """ImageNet dataset used for ILSVRC2012 [1].

    .. [1] ImageNet Large Scale Visual Recognition Challenge
        http://arxiv.org/abs/1409.0575

    If you pass `\'auto\'` as an argument for `base_dir`, this directory
    tries to download from `urls`. If `urls` is `None` in that case, it will
    look up for dataset in the filesystem, but do not download anything.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            'auto', this class will automatically download data for you
            under ``$CHAINER_DATASET_ROOT/pfnet/chainer_cv/imagenet``.
        urls (list of strings): the list contains four urls of
            `[{Train images Tasks 1 & 2}, {Train images Task 3},
            {Validation Images}, {Test Images}]`.

    """

    def __init__(self, data_dir='auto', urls=None, use_cache=False, delete_cache=False):
        if data_dir == 'auto':
            data_dir = _get_imagenet(urls)
        self.data_dir = data_dir

        self.synset_map = get_imagenet_synset_map()
        train_fns_pkl = osp.join(
            self.data_dir, 'train_image_files.pkl')
        self.train_fns = cache_load(
            train_fns_pkl, self._get_train_image_files, delete_cache,
            use_cache,
            args=(osp.join(data_dir, 'train'),))

    def _get_train_image_files(self, train_dir):
        image_fns = {}
        for fn in os.listdir(train_dir):
            synset = fn[:9]
            if synset in self.synset_map and fn[-4:] == 'JPEG':
                if synset not in image_fns:
                    image_fns[synset] = []
                image_fns[synset].append(osp.join(train_dir, fn))
        return image_fns

    def _get_val_test_image_files(self, data_dir, mode):
        if mode not in ['val', 'test']:
            raise ValueError
        image_fns = {}
        prefix = 'ILSVRC2012_{}'.format(mode)
        for fn in os.listdir(data_dir):
            if prefix in fn and fn[-4:] == 'JPEG':
                image_fns


if __name__ == '__main__':
    urls = {
        'train': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'val': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'test': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar',
    }
    dataset = ImagenetDataset(urls=urls)
