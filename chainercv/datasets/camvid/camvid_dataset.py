import glob
import os
import shutil

import numpy as np

import chainer
from chainer.dataset import download
from chainercv import utils
from chainercv.utils import read_image


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


class CamVidDataset(chainer.dataset.DatasetMixin):

    """Dataset class for a semantic segmantion task on CamVid `u`_.

    .. _`u`: https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/camvid`.
        split ({'train', 'val', 'test'}): Select from dataset splits used
            in CamVid Dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError(
                'Please pick split from \'train\', \'val\', \'test\'')

        if data_dir == 'auto':
            data_dir = get_camvid()

        img_list_filename = os.path.join(data_dir, '{}.txt'.format(split))
        self.filenames = [
            [os.path.join(data_dir, fn.replace('/SegNet/CamVid/', ''))
             for fn in line.split()] for line in open(img_list_filename)]

    def __len__(self):
        return len(self.filenames)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. The color image is in CHW
        format and the label image is in HW format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of a color image and a label whose shapes are (3, H, W) and
            (H, W) respectively. H and W are height and width of the image.
            The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        img_filename, label_filename = self.filenames[i]
        img = read_image(img_filename, color=True)
        label = read_image(label_filename, dtype=np.int32, color=False)[0]
        # Label id 11 is for unlabeled pixels.
        label[label == 11] = -1
        return img, label
