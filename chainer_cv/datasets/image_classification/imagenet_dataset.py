import tarfile

import chainer
from chainer.dataset import download

from chainer_cv import utils


root = 'pfnet/chainer_cv/imagenet'


def _get_imagenet(urls):
    data_root = download.get_dataset_directory(root)
    # base_path = osp.join(data_root, 'Stanford_Online_Products')
    # if osp.exists(base_path):
    #     return base_path

    for url in urls:
        download_file_path = utils.cached_download(url)

        with tarfile.TarFile(download_file_path, 'r') as t:
            t.extractall(data_root)
    # return base_path


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

    def __init__(self, data_dir='auto', urls=None):
        if data_dir == 'auto':
            data_dir = _get_imagenet(urls)
        self.data_dir = data_dir


if __name__ == '__main__':
    urls = ''
    ImagenetDataset(urls=urls)
