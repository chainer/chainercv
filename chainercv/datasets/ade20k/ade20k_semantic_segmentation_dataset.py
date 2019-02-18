import glob
import os

import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.ade20k.ade20k_utils import get_ade20k
from chainercv.utils import read_image
from chainercv.utils import read_label

root = 'pfnet/chainercv/ade20k'
url = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


class ADE20KSemanticSegmentationDataset(GetterDataset):

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

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    """

    def __init__(self, data_dir='auto', split='train'):
        super(ADE20KSemanticSegmentationDataset, self).__init__()

        if data_dir is 'auto':
            data_dir = get_ade20k(root, url)

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

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        return read_image(self.img_paths[i])

    def _get_label(self, i):
        label = read_label(self.label_paths[i], dtype=np.int32)
        # [-1, n_class - 1]
        return label - 1
