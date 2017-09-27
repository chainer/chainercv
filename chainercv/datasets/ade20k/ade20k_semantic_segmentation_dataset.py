import glob
import os

import numpy as np

from chainer import dataset
from chainer.dataset import download
from chainercv import utils
from chainercv.utils import read_image

root = 'pfnet/chainercv/ade20k'
url = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def get_ade20k():
    data_root = download.get_dataset_directory(root)
    cache_fn = utils.cached_download(url)
    utils.extractall(cache_fn, data_root, os.path.splitext(url)[1])
    return data_root


class ADE20KSemanticSegmentationDataset(dataset.DatasetMixin):

    """Semantic segmentation dataset for `ADE20K`_.

    This is ADE20K dataset distributed in MIT Scene Parsing Benchmark website.
    It has 20,210 training images and 2,000 validation images.

    .. _`MIT Scene Parsing Benchmark`: http://sceneparsing.csail.mit.edu/

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain the :obj:`ADEChallengeData2016` directory. And that
            directory should contain at least :obj:`images` and
            :obj:`annotations` directries. If :obj:`auto` is given, the dataset
            is automatically downloaded into
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/ade20k`.
        split ({'train', 'val'}): Select from dataset splits used in
            MIT Scene Parsing Benchmark dataset (ADE20K).

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
        else:
            raise ValueError(
                'Please give \'split\' argument with either \'train\' or '
                '\'val\'.')

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            Returns a tuple consited of a color image and a label whose shapes
            are (3, H, W) and (H, W), respectively. H and W are height and
            width of the image. The dtype of the color image is
            :obj:`numpy.float32` and the dtype of the label image is
            :obj:`numpy.int32`.

        """
        img = read_image(self.img_paths[i])
        label = read_image(self.label_paths[i], dtype=np.int32, color=False)[0]
        return img, label
