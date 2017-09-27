import glob
from multiprocessing import Pool
import os

import numpy as np

from chainer import dataset
from chainer.dataset import download
from chainercv import utils
from chainercv.utils import read_image

root = 'pfnet/chainercv/ade20k'
trainval_url = 'http://data.csail.mit.edu/places/ADEchallenge/'
trainval_url += 'ADEChallengeData2016.zip'
test_url = 'http://data.csail.mit.edu/places/ADEchallenge/release_test.zip'


def get_ade20k():
    p = Pool(2)
    data_root = download.get_dataset_directory(root)
    urls = [trainval_url, test_url]
    ret = [p.apply_async(utils.cached_download, args=(url,)) for url in urls]
    caches = [r.get() for r in ret]
    args = [(cache_fn, data_root, os.path.splitext(url)[1])
            for cache_fn, url in zip(caches, urls)]
    ret = [p.apply_async(utils.extractall, args=arg) for arg in args]
    for r in ret:
        r.get()
    return data_root


class ADE20KSemanticSegmentationDataset(dataset.DatasetMixin):

    """Semantic segmentation dataset for `ADE20K`_.

    This is ADE20K dataset distributed in MIT Scene Parsing Benchmark website.
    It has 20,210 training images, 2,000 validation images, and 3,352 test
    images.

    .. _`MIT Scene Parsing Benchmark`: http://sceneparsing.csail.mit.edu/

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least two directories, :obj:`annotations` and
            :obj:`images`. If :obj:`auto` is given, the dataset is
            automatically downloaded into
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/ade20k`.
        split ({'train', 'val', 'test'}): Select from dataset splits used in
            Cityscapes dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if data_dir is 'auto':
            data_dir = get_ade20k()

        if split == 'train' or split == 'val':
            img_dir = os.path.join(
                data_dir, 'ADEChallengeData2016', 'images',
                'training' if split == 'train' else 'validation')
            label_dir = os.path.join(
                data_dir, 'ADEChallengeData2016', 'annotations',
                'training' if split == 'train' else 'validation')
        elif split == 'test':
            img_dir = os.path.join(data_dir, 'release_test', 'testing')
        else:
            raise ValueError(
                'Please give \'split\' argument with either \'train\', '
                '\'val\', or \'test\'.')

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if split == 'train' or split == 'val':
            self.label_paths = sorted(
                glob.glob(os.path.join(label_dir, '*.png')))

        self.split = split

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            When :obj:`split` is either :obj:`train` or :obj:`val`, it returns
            a tuple consited of a color image and a label whose shapes are
            (3, H, W) and (H, W), respectively, while :obj:`split` is
            :obj:`test`, it returns only the color image. H and W are height
            and width of the image. The dtype of the color image is
            :obj:`numpy.float32` and the dtype of the label image is
            :obj:`numpy.int32`.

        """
        img = read_image(self.img_paths[i])
        if self.split == 'train' or self.split == 'val':
            label = read_image(
                self.label_paths[i], dtype=np.int32, color=False)[0]
            return img, label
        else:
            return img
