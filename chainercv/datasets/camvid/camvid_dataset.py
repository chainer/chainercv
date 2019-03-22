import filelock
import glob
import os
import shutil

import numpy as np

from chainer.dataset import download

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv import utils
from chainercv.utils import read_image
from chainercv.utils import read_label


root = 'pfnet/chainercv/camvid'
url = 'https://github.com/alexgkendall/SegNet-Tutorial/archive/master.zip'

# https://github.com/alexgkendall/SegNet-Tutorial/blob/master/
# Scripts/test_segmentation_camvid.py#L62
camvid_label_names = (
    'Sky',
    'Building',
    'Pole',
    'Road',
    'Pavement',
    'Tree',
    'SignSymbol',
    'Fence',
    'Car',
    'Pedestrian',
    'Bicyclist',
)

camvid_label_colors = (
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (60, 40, 222),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
)
camvid_ignore_label_color = (0, 0, 0)


def get_camvid():
    # To support ChainerMN, the target directory should be locked.
    with filelock.FileLock(os.path.join(download.get_dataset_directory(
            'pfnet/chainercv/.lock'), 'camvid.lock')):
        data_root = download.get_dataset_directory(root)
        download_file_path = utils.cached_download(url)
        if len(glob.glob(os.path.join(data_root, '*'))) != 9:
            utils.extractall(
                download_file_path, data_root, os.path.splitext(url)[1])
        data_dir = os.path.join(data_root, 'SegNet-Tutorial-master/CamVid')
        if os.path.exists(data_dir):
            for fn in glob.glob(os.path.join(data_dir, '*')):
                shutil.move(fn, os.path.join(data_root, os.path.basename(fn)))
            shutil.rmtree(os.path.dirname(data_dir))
    return data_root


class CamVidDataset(GetterDataset):

    """Semantic segmentation dataset for `CamVid`_.

    .. _`CamVid`:
        https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/camvid`.
        split ({'train', 'val', 'test'}): Select from dataset splits used
            in CamVid Dataset.


    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    """

    def __init__(self, data_dir='auto', split='train'):
        super(CamVidDataset, self).__init__()

        if split not in ['train', 'val', 'test']:
            raise ValueError(
                'Please pick split from \'train\', \'val\', \'test\'')

        if data_dir == 'auto':
            data_dir = get_camvid()

        img_list_path = os.path.join(data_dir, '{}.txt'.format(split))
        self.paths = [
            [os.path.join(data_dir, fn.replace('/SegNet/CamVid/', ''))
             for fn in line.split()] for line in open(img_list_path)]

        self.add_getter('img', self._get_image)
        self.add_getter('iabel', self._get_label)

    def __len__(self):
        return len(self.paths)

    def _get_image(self, i):
        img_path, _ = self.paths[i]
        return read_image(img_path, color=True)

    def _get_label(self, i):
        _, label_path = self.paths[i]
        label = read_label(label_path, dtype=np.int32)
        # Label id 11 is for unlabeled pixels.
        label[label == 11] = -1
        return label
