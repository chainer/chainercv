import numpy as np
import os.path as osp
import glob
import shutil
import chainer
from chainer.dataset import download
from chainercv import utils

from chainercv.utils import read_image


root = 'pfnet/chainercv/camvid'
url = 'https://github.com/alexgkendall/SegNet-Tutorial/archive/master.zip'


def get_camvid():
    data_root = download.get_dataset_directory(root)
    download_file_path = utils.cached_download(url)
    if len(glob.glob(osp.join(data_root, '*'))) != 9:
        utils.extractall(download_file_path, data_root, osp.splitext(url)[1])
    data_dir = osp.join(data_root, 'SegNet-Tutorial-master/CamVid')
    if osp.exists(data_dir):
        for fn in glob.glob(osp.join(data_dir, '*')):
            shutil.move(fn, osp.join(data_root, osp.basename(fn)))
        shutil.rmtree(osp.dirname(data_dir))
    return data_root


class CamVidDataset(chainer.dataset.DatasetMixin):

    """Dataset class for a semantic segmantion task on CamVid `CamVid`_.

    .. _`CamVid`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/pascal_voc`.
        mode ({'train', 'val', 'trainval'}): Select from dataset splits used
            in VOC.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If true, use images that are labeled as
            difficult in the original annotation.

    """

    def __init__(self, data_dir='auto', mode='train'):
        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                'Please pick mode from \'train\', \'val\', \'test\'')

        if data_dir == 'auto':
            data_dir = get_camvid()

        img_list_fn = osp.join(data_dir, '{}.txt'.format(mode))
        self.fns = [[osp.join(data_dir, fn.replace('/SegNet/CamVid/', ''))
                     for fn in line.split()] for line in open(img_list_fn)]

    def __len__(self):
        return len(self.fns)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. Both of them are in CHW
        format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of color image and label whose shapes are (3, H, W) and
            (1, H, W) respectively. H and W are height and width of the
            images. The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        image_fn, label_fn = self.fns[i]
        image = read_image(image_fn, color=True)
        label = read_image(label_fn, dtype=np.int32, color=False)
        return image, label
